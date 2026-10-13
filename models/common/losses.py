"""Masked regression losses over an action chunk.

Both policies predict a fixed-length chunk of future actions. Near the end of an episode
the chunk runs past the last step, so the dataset fills the tail and marks it with
`is_pad`. Both policies need the same "average the error, but ignore the padding"
arithmetic, so it lives here once instead of in each of them.

The grid of shapes every caller uses:

    prediction, target: (batch, chunk, action_dim)
    is_pad:             (batch, chunk) bool, True on the padded steps
"""

import torch.nn.functional as F


def masked_mean(error, is_pad):
    """Average a per-element error over the valid (non-padding) entries.

    The denominator counts valid *elements*, not valid steps, hence the `* action_dim`.
    `.clamp(min=1)` keeps an all-padding chunk from dividing by zero and returning NaN.
    """
    valid = ~is_pad.unsqueeze(-1)                      # (batch, chunk, 1), broadcasts over action_dim
    denominator = (valid.sum() * error.size(-1)).clamp(min=1)
    return (error * valid).sum() / denominator


def masked_mse_loss(prediction, target, is_pad):
    """Masked mean squared error.

    The flow-matching expert uses this: its target is the velocity, not the action, and
    velocity is roughly symmetric around zero so there is no mode structure for L1 to
    protect.
    """
    return masked_mean((prediction - target) ** 2, is_pad)


def masked_l1_loss(prediction, target, is_pad):
    """Masked mean absolute error.

    The regression baseline uses this. L1 rather than MSE because the gripper dimension
    is nearly bimodal - open or closed, rarely in between - and the mean of two modes is
    a command the robot is never asked to hold.
    """
    return masked_mean(F.l1_loss(prediction, target, reduction="none"), is_pad)
