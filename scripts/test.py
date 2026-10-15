"""Evaluate a trained policy in LIBERO.

Run it as a module, from the project root, inside the `lerobot` conda env:

    python -m scripts.test --checkpoint checkpoints/smolvla_goal.pt --episodes 5

The checkpoint carries the config it was trained with, so no config file is needed here.

For every task one success and one failure are recorded (a task that never fails, or never
succeeds, simply has only the one kind). The file name says which task it was:

    videos/task03_open_the_top_drawer_and_put_the_bowl_inside_start02_success.mp4
"""

import argparse
import os
import sys
import time

import imageio.v2 as imageio
import torch

from data.libero_dataset import load_tokenizer, suite_languages, tokenize_instructions
from envs.libero_env import LiberoEnvironment, max_steps
from models.build import build_policy
from models.common.checkpoint import load_trainable_state_dict

# print() is block-buffered when the output goes to a file, which would hide the per-task
# results until the whole evaluation finished. Flush every line instead.
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a policy in LIBERO")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5,
                        help="start states per task, out of the 50 LIBERO provides")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True,
                        help="record successes and failures of every task")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--video-per-outcome", type=int, default=1,
                        help="videos kept per task for each outcome (success and fail)")
    parser.add_argument("--shuffle-instruction", action="store_true",
                        help="give every task another task's instruction")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_policy(checkpoint_path, device):
    """Rebuild the policy from the config and statistics the checkpoint carries."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    run_config = checkpoint["config"]
    policy = build_policy(run_config, checkpoint["norm_stats"])
    load_trainable_state_dict(policy, checkpoint["model_state_dict"])
    policy = policy.to(device)
    policy.eval()

    print(f"[test] policy={run_config['policy']}  suite={run_config['data']['suite']}  "
          f"camera_size={run_config['data']['camera_size']}  "
          f"chunk={run_config['data']['chunk_size']}")
    return policy, run_config


def make_batch(images, state, instruction, device):
    """The one-sample batch the policy's act() expects."""
    text_ids, text_mask = instruction
    return {
        "images": images.unsqueeze(0).to(device),
        "state": state.unsqueeze(0).to(device),
        "text_ids": text_ids.unsqueeze(0).to(device),
        "text_mask": text_mask.unsqueeze(0).to(device),
    }


def run_episode(env, policy, task_id, start_index, instruction, device, step_limit):
    """Run one episode. Returns (success, steps, frames)."""
    images, state, instruction_from_env = env.reset(task_id, start_index)

    frames = []
    action_queue = []
    steps = 0
    success = False

    # the step limit is enforced here rather than trusted to `done`, which LIBERO also sets
    # when the episode merely runs out of time
    while steps < step_limit:
        if not action_queue:
            chunk = policy.act(make_batch(images, state, instruction, device))
            # one action per environment step; re-plan when the queue runs dry
            action_queue = list(chunk.squeeze(0).cpu().numpy())

        action = action_queue.pop(0)
        images, state, reward, success, done = env.step(action)
        steps += 1
        frames.append(images)
        if success:
            break

    return success, steps, frames, instruction_from_env


def save_video(frames, path):
    """Write the recorded frames to an mp4, both cameras side by side."""
    with imageio.get_writer(path, fps=10) as writer:     # the demonstrations are 10 fps
        for images in frames:
            cameras, channels, height, width = images.shape
            # (cameras, C, H, W) -> (C, H, cameras, W) -> (C, H, cameras * W), so the two
            # cameras end up next to each other horizontally
            side_by_side = images.permute(1, 2, 0, 3)
            side_by_side = side_by_side.reshape(channels, height, cameras * width)
            frame = side_by_side.permute(1, 2, 0).cpu().numpy()
            writer.append_data((frame * 255).astype("uint8"))


def video_name(task_id, real_instruction, asked_for, start_index, outcome):
    """A file name that says which task the scene was.

    With --shuffle-instruction the policy is told a different instruction from the scene's
    own, so that one is added too; otherwise the video would look like the policy failed the
    task it was named after.
    """
    name = f"task{task_id:02d}_{real_instruction.replace(' ', '_')}"
    if asked_for != real_instruction:
        name += f"_told_{asked_for.replace(' ', '_')}"
    return f"{name}_start{start_index:02d}_{outcome}.mp4"


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy, run_config = load_policy(args.checkpoint, device)

    suite_name = run_config["data"]["suite"]
    camera_size = run_config["data"]["camera_size"]
    step_limit = max_steps(suite_name)

    env = LiberoEnvironment(suite_name, camera_size)

    # every instruction of the suite, tokenised once. Looking the instruction up in this
    # is also what makes --shuffle-instruction a two-line change.
    languages = suite_languages(suite_name)
    instructions = tokenize_instructions(load_tokenizer(), languages)
    if args.shuffle_instruction:
        print("[test] instructions shuffled: each task is given the next task's instruction")

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    total_successes = 0
    total_episodes = 0
    started = time.time()

    try:
        for task_id in range(env.task_count()):
            real_instruction = languages[task_id]
            asked_for = real_instruction
            if args.shuffle_instruction:
                asked_for = languages[(task_id + 1) % len(languages)]

            task_successes = 0
            videos_of_task = {"success": 0, "fail": 0}    # counted per task, not overall
            for start_index in range(args.episodes):
                success, steps, frames, _ = run_episode(
                    env, policy, task_id, start_index, instructions[asked_for],
                    device, step_limit)
                task_successes += int(success)

                outcome = "success" if success else "fail"
                if args.save_video and videos_of_task[outcome] < args.video_per_outcome:
                    videos_of_task[outcome] += 1
                    name = video_name(task_id, real_instruction, asked_for, start_index,
                                      outcome)
                    save_video(frames, os.path.join(args.video_dir, name))

            total_successes += task_successes
            total_episodes += args.episodes
            print(f"task {task_id:2d}  {task_successes}/{args.episodes}  "
                  f"({100 * task_successes / args.episodes:5.1f}%)  asked: {asked_for}")

            # what the task actually was, when the two differ
            if asked_for != real_instruction:
                print(f"            (the scene's real task: {real_instruction})")
    finally:
        env.close()

    print(f"\noverall {total_successes}/{total_episodes} = "
          f"{100 * total_successes / total_episodes:.1f}%  "
          f"({(time.time() - started) / 60:.1f} min)")
    if args.shuffle_instruction:
        print("these are the shuffled numbers; compare them against a run without the flag")


if __name__ == "__main__":
    main()
