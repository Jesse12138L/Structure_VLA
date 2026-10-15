"""Train a policy from a YAML config in configs/.

The config defines the experiment; the command line can override a few of its scalars but
cannot add a new setting. The whole config goes into the checkpoint, so scripts/test.py can
rebuild the model without needing the config file.

Run it as a module, from the project root, inside the `lerobot` conda env:

    python -m scripts.train --config configs/smolvla_goal.yaml

A first run that has to prove the plumbing works, without waiting for a real training run:

    python -m scripts.train --config configs/smolvla_goal.yaml --overfit-task 8 --epochs 30

Every epoch is saved, next to the save path in a folder of the same name:

    checkpoints/smolvla_goal/epoch_001.pt, epoch_002.pt, ...

and the epoch with the lowest validation loss is also copied to the save path itself,
checkpoints/smolvla_goal.pt, which is the file scripts/test.py is normally pointed at.
"""

import argparse
import os
import shutil
import sys
import time

import torch
from torch.utils.data import DataLoader

from data.libero_dataset import build as build_data
from models.build import build_policy
from models.common.checkpoint import trainable_state_dict
from utils.config import load, with_overrides
from utils.ema import make_ema

# print() is block-buffered when the output goes to a file, which would hide the progress
# of a multi-hour run until it finished. Flush every line instead.
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a policy on the LIBERO demonstrations")
    parser.add_argument("--config", type=str, required=True)
    # run-level overrides. Each one has to name a key the YAML already has, so a typo in a
    # flag cannot quietly introduce a setting nothing else knows about.
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="override train.lr")
    parser.add_argument("--device", type=str, default=None, help="override train.device")
    parser.add_argument("--camera-size", type=int, default=None, help="override data.camera_size")
    parser.add_argument("--chunk-size", type=int, default=None, help="override data.chunk_size")
    parser.add_argument("--overfit-task", type=int, default=None,
                        help="override data.overfit_task: a LIBERO task id 0-9 in the suite")
    parser.add_argument("--save-path", type=str, default=None,
                        help="override train.save_path, so a test run cannot overwrite a real one")
    return parser.parse_args()


def overrides_from(args):
    """The command-line values as dotted config paths. None means 'flag not given'."""
    return {
        "train.epochs": args.epochs,
        "train.batch_size": args.batch_size,
        "train.lr": args.lr,
        "train.device": args.device,
        "train.save_path": args.save_path,
        "data.camera_size": args.camera_size,
        "data.chunk_size": args.chunk_size,
        "data.overfit_task": args.overfit_task,
    }


def to_device(batch, device):
    """Move every tensor in a batch to the device. The instruction strings stay where they are."""
    moved = {}
    for key in batch:
        value = batch[key]
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def trainable_parameters(policy):
    """The parameters to optimise: everything except the frozen trunk."""
    return [p for p in policy.parameters() if p.requires_grad]


def save_checkpoint(save_path, policy, run_config, norm_stats):
    """Write the trainable weights, the config and the statistics to disk.

    Only the trainable part is stored: the trunk is frozen, and scripts/test.py rebuilds it
    from its name. The config and the statistics travel inside the checkpoint, so evaluation
    needs no YAML.
    """
    torch.save(
        {
            "model_state_dict": trainable_state_dict(policy),
            "config": run_config,
            "norm_stats": norm_stats,
        },
        save_path,
    )


def evaluate(policy, loader, device):
    """Run the whole validation set and return two numbers.

    The first is the model's average loss. The second is the loss of a model that has
    learned nothing: it ignores the observation and always gives the same answer. The
    first number means nothing on its own - read it next to the second, because a
    validation loss above it means the model does worse than ignoring the camera and the
    instruction.

    Both numbers move a little from call to call, since the flow matching loss draws new
    noise and time each time. Over thousands of frames that difference is small.
    """
    policy.eval()
    total_loss = 0.0
    total_reference = 0.0
    for batch in loader:
        moved = to_device(batch, device)
        with torch.no_grad():
            count = moved["images"].size(0)
            total_loss += float(policy.loss(moved)) * count
            total_reference += float(policy.reference_loss(moved)) * count
    frames = len(loader.dataset)
    return total_loss / frames, total_reference / frames


def main():
    args = parse_args()
    run_config = with_overrides(load(args.config), overrides_from(args))# args.config yaml文件的参数

    data_config = run_config["data"]
    train_config = run_config["train"]
    save_path = train_config["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # checkpoints/smolvla_goal.pt -> checkpoints/smolvla_goal/, where every epoch is kept
    epochs_dir = os.path.splitext(save_path)[0]
    os.makedirs(epochs_dir, exist_ok=True)

    if not torch.cuda.is_available() and train_config["device"] == "cuda":
        print("[warn] cuda is not available, falling back to cpu")
    device = torch.device(train_config["device"] if torch.cuda.is_available() else "cpu")

    started = time.time()
    train_samples, val_samples, norm_stats = build_data(run_config)
    print(f"[data] train {len(train_samples)} frames, val {len(val_samples)} frames, "
          f"built in {time.time() - started:.0f}s")
    if data_config["overfit_task"] is not None:
        print(f"[data] narrowed to suite task {data_config['overfit_task']}, "
              f"for the overfit check")

    train_loader = DataLoader(train_samples, batch_size=train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=train_config["batch_size"])

    policy = build_policy(run_config, norm_stats).to(device)
    trained_parameters = trainable_parameters(policy)
    frozen_count = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
    trained_count = sum(p.numel() for p in trained_parameters)
    print(f"[model] policy={run_config['policy']}  frozen={frozen_count/1e6:.1f}M  "
          f"trainable={trained_count/1e6:.1f}M  device={device}")

    optimizer = torch.optim.AdamW(trained_parameters, lr=train_config["lr"])
    ema = None
    if train_config["use_ema"]:
        ema = make_ema(trained_parameters, power=train_config["ema_power"])

    best_val = float("inf")
    best_epoch = -1
    for epoch in range(train_config["epochs"]):
        policy.train()
        train_total = 0.0
        for batch in train_loader:
            loss = policy.loss(to_device(batch, device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.step(trained_parameters)
            train_total += float(loss.detach()) * batch["images"].size(0)

        # Evaluate and save under the same weights. With EMA those are the averaged ones, so
        # the checkpoint has to be written before the live weights are put back; otherwise
        # it would hold weights that were never the ones the validation loss was measured on.
        saved = None
        if ema is not None:
            saved = ema.swap_in(trained_parameters)
        val_loss, reference = evaluate(policy, val_loader, device)
        epoch_path = os.path.join(epochs_dir, f"epoch_{epoch + 1:03d}.pt")
        save_checkpoint(epoch_path, policy, run_config, norm_stats)
        if ema is not None:
            ema.swap_out(trained_parameters, saved)

        # Every epoch is kept, because the validation loss and the closed-loop success rate
        # do not always agree on which epoch is best. The lowest-validation epoch is also
        # copied to save_path, so evaluation can use one fixed file name. Written out as the
        # run goes, so a run that is killed or loses its container at hour six still leaves
        # everything it had found.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            shutil.copyfile(epoch_path, save_path)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:4d}/{train_config['epochs']}  "
                  f"train={train_total / len(train_samples):.4f}  "
                  f"val={val_loss:.4f}  no-observation={reference:.4f}  "
                  f"best={best_val:.4f}@{best_epoch}  "
                  f"({(time.time() - started) / 60:.1f} min)")

    print(f"best val {best_val:.4f} at epoch {best_epoch}; {save_path} holds the weights from "
          f"that epoch, and {epochs_dir}/ holds every epoch")


if __name__ == "__main__":
    main()
