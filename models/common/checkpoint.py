"""Saving and loading only the parameters that were trained.

The frozen trunk is about 303M parameters that come back identical from HuggingFace on
every load. Storing them would make every checkpoint roughly twenty times larger for no
benefit, so a checkpoint holds everything except `backbone.*` - which is the expert plus
the normalisation statistics, a few tens of megabytes.
"""


def trainable_state_dict(policy):
    """The policy's state, minus the frozen backbone."""
    state = {}
    for name, value in policy.state_dict().items():
        if name.startswith("backbone."):
            continue
        state[name] = value
    return state


def load_trainable_state_dict(policy, state):
    """Load a checkpoint written by `trainable_state_dict`.

    The key sets are compared in both directions before loading. `load_state_dict` would
    otherwise accept a checkpoint from a differently shaped model without complaint, and
    the result would be a policy that runs but predicts nonsense.
    """
    expected = set(trainable_state_dict(policy).keys())
    found = set(state.keys())
    missing = sorted(expected - found)
    if missing:
        raise KeyError(f"checkpoint is missing these parameters: {missing}")
    unexpected = sorted(found - expected)
    if unexpected:
        raise KeyError(f"checkpoint has parameters this model does not use: {unexpected}")

    # strict=False only because the backbone is deliberately absent from the checkpoint
    policy.load_state_dict(state, strict=False)
