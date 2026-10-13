"""The flow matching policy: a frozen SmolVLM2 trunk plus the denoising action expert.

This is the "SmolVLA" of this project. The two things it owns that the trunk and the
expert do not are the normalisation and the loss/inference contract:

    loss(batch)  one training step's loss
    act(batch)   a full action chunk, sampled from noise

Normalisation lives here rather than in the dataset so that the statistics are stored in
the checkpoint and act() can undo them. Actions and states are z-scored: the gripper
dimension of a LIBERO action is nearly +-1 while its rotation dimensions have a standard
deviation under 0.1, and without scaling the loss would be almost entirely about the
gripper.
"""

import torch
import torch.nn as nn

from ..common.flow_matching import (make_noisy_actions, sample_euler,
                                    sample_training_time, velocity_target)
from ..common.losses import masked_mse_loss


class SmolVLAFlowPolicy(nn.Module):
    """Frozen trunk + flow matching expert, with the normalisation around them."""

    def __init__(self, backbone, expert, norm_stats, chunk_size, action_dim,
                 num_flow_steps=10):
        super().__init__()
        self.backbone = backbone
        self.expert = expert
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.num_flow_steps = num_flow_steps

        # buffers, not plain attributes, so they follow the model between devices and are
        # part of the checkpoint
        self.register_buffer("state_mean", norm_stats["state_mean"])
        self.register_buffer("state_std", norm_stats["state_std"])
        self.register_buffer("action_mean", norm_stats["action_mean"])
        self.register_buffer("action_std", norm_stats["action_std"])

    def encode(self, batch):
        """The observation and instruction as prefix tokens, plus the normalised state."""
        prefix, prefix_mask = self.backbone(batch["images"], batch["text_ids"],
                                           batch["text_mask"])
        # the trunk runs in bfloat16, but the expert is trained in float32: its random
        # noise and its MSE loss both want the extra precision
        prefix = prefix.float()
        state = (batch["state"] - self.state_mean) / self.state_std
        return prefix, prefix_mask, state

    def loss(self, batch):
        """One training step's loss. `batch` is what data/libero_dataset.py returns."""
        prefix, prefix_mask, state = self.encode(batch)
        action = (batch["action"] - self.action_mean) / self.action_std

        noise = torch.randn_like(action)
        time = sample_training_time(action.shape[0], action.device)
        noisy_action = make_noisy_actions(action, noise, time)

        velocity = self.expert(prefix, prefix_mask, noisy_action, time, state,
                              batch["action_is_pad"])
        return masked_mse_loss(velocity, velocity_target(action, noise),
                               batch["action_is_pad"])

    def reference_loss(self, batch):
        """What the loss would be with no observation at all: predicting zero velocity.

        The velocity target has mean zero, so this is the score of the model that has
        learned nothing. Printed beside the validation loss, it is the first thing to check
        - a validation loss above this means the expert is worse than useless.
        """
        action = (batch["action"] - self.action_mean) / self.action_std
        noise = torch.randn_like(action)
        no_prediction = torch.zeros_like(action)
        return masked_mse_loss(no_prediction, velocity_target(action, noise),
                               batch["action_is_pad"])

    @torch.no_grad()
    def act(self, batch):
        """A full action chunk, denoised from noise and returned in real action units."""
        prefix, prefix_mask, state = self.encode(batch)
        batch_size = prefix.shape[0]
        device = prefix.device

        noise = torch.randn(batch_size, self.chunk_size, self.action_dim, device=device)
        # nothing is padded at inference: every action of the chunk is about to be used
        no_padding = torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=device)

        def denoise(noisy_action, time):
            return self.expert(prefix, prefix_mask, noisy_action, time, state, no_padding)

        normalized = sample_euler(denoise, noise, self.num_flow_steps)
        return normalized * self.action_std + self.action_mean
