"""Build a policy from a config dict.

Used by both scripts/train.py and scripts/test.py, so the model a checkpoint is rebuilt
from at evaluation is necessarily the one that was trained.

Config schema (see configs/*.yaml). Only the blocks the chosen policy reads are used:

    policy: "smolvla" | "baseline"

    data:
      chunk_size: int             actions predicted per forward pass

    model:
      trunk_layers: int           how many of SmolVLM2's 32 text layers to keep
      smolvla:  { d_exp, n_exp_layers, n_heads, d_ff, dropout, num_flow_steps }
      baseline: { d_head, n_heads, d_ff, dropout }

There is no schema validation beyond the unknown-policy case: a missing key raises a plain
KeyError naming it, as in mini-vla.
"""

from .backbones.smolvlm2 import SmolVLM2Backbone
from .experts.flow_expert import FlowActionExpert
from .experts.regression import RegressionActionExpert
from .policies.baseline_policy import BaselinePolicy
from .policies.smolvla_policy import SmolVLAFlowPolicy


def build_backbone(model_config):
    """The frozen trunk. One implementation today; a new one is a new branch here."""
    return SmolVLM2Backbone(num_layers=model_config["trunk_layers"])


def build_policy(config, norm_stats):
    """Instantiate the policy named by config["policy"], still on the CPU.

    `norm_stats` comes from the dataset while training and from the checkpoint while
    evaluating - never from the config, which is why it is an argument rather than a key.
    The state and action sizes are read off those statistics instead of being passed
    separately, so the model cannot disagree with the data about them.

    Note that with a fresh trunk every policy starts from the same frozen features, so two
    runs with the same config and seed differ only through the expert's initialisation.
    """
    data = config["data"]
    model = config["model"]
    chunk_size = data["chunk_size"]

    # the state and action means are (state_dim,) and (action_dim,) vectors
    state_dim = norm_stats["state_mean"].shape[0]
    action_dim = norm_stats["action_mean"].shape[0]

    backbone = build_backbone(model)
    prefix_size = backbone.hidden_size

    if config["policy"] == "smolvla":
        settings = model["smolvla"]
        expert = FlowActionExpert(
            prefix_size=prefix_size,
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_exp=settings["d_exp"],
            n_heads=settings["n_heads"],
            n_layers=settings["n_exp_layers"],
            d_ff=settings["d_ff"],
            dropout=settings["dropout"],
        )
        return SmolVLAFlowPolicy(backbone, expert, norm_stats, chunk_size, action_dim,
                                 num_flow_steps=settings["num_flow_steps"])

    if config["policy"] == "baseline":
        settings = model["baseline"]
        expert = RegressionActionExpert(
            prefix_size=prefix_size,
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            d_head=settings["d_head"],
            n_heads=settings["n_heads"],
            d_ff=settings["d_ff"],
            dropout=settings["dropout"],
        )
        return BaselinePolicy(backbone, expert, norm_stats, chunk_size)

    raise ValueError(f"unknown policy {config['policy']!r}; expected 'smolvla' or 'baseline'")
