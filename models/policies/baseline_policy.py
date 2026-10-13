"""The baseline policy: the same frozen trunk, but the action chunk in one shot.

The trunk, the normalisation and the loss/inference contract are identical to
SmolVLAFlowPolicy - only the action expert differs, and it is the regression head rather
than the denoising one. That is the whole point: run this against the flow matching policy
with everything else held fixed, and whatever separates them came from the flow matching.
"""

import torch
import torch.nn as nn

from ..common.losses import masked_l1_loss


class BaselinePolicy(nn.Module):
    """Frozen trunk + regression head, with the normalisation around them."""

    def __init__(self, backbone, expert, norm_stats, chunk_size):
        super().__init__()
        self.backbone = backbone
        self.expert = expert
        self.chunk_size = chunk_size

        self.register_buffer("state_mean", norm_stats["state_mean"])
        self.register_buffer("state_std", norm_stats["state_std"])
        self.register_buffer("action_mean", norm_stats["action_mean"])
        self.register_buffer("action_std", norm_stats["action_std"])

    def encode(self, batch):
        """The observation and instruction as prefix tokens, plus the normalised state."""
        prefix, prefix_mask = self.backbone(batch["images"], batch["text_ids"],
                                           batch["text_mask"])
        prefix = prefix.float()
        state = (batch["state"] - self.state_mean) / self.state_std
        return prefix, prefix_mask, state

    def loss(self, batch):
        """One training step's loss. `batch` is what data/libero_dataset.py returns."""
        prefix, prefix_mask, state = self.encode(batch)
        action = (batch["action"] - self.action_mean) / self.action_std
        predicted = self.expert(prefix, prefix_mask, state)
        return masked_l1_loss(predicted, action, batch["action_is_pad"])

    def reference_loss(self, batch):
        """What the loss would be with no observation at all.

        The actions are z-scored, so their mean is zero: predicting the training mean means
        predicting zeros. This is the score of the model that has learned nothing, and a
        validation loss above it means the head is worse than useless.
        """
        action = (batch["action"] - self.action_mean) / self.action_std
        no_prediction = torch.zeros_like(action)
        return masked_l1_loss(no_prediction, action, batch["action_is_pad"])

    @torch.no_grad()
    def act(self, batch):
        """A full action chunk in one forward pass, returned in real action units."""
        prefix, prefix_mask, state = self.encode(batch)
        normalized = self.expert(prefix, prefix_mask, state)
        return normalized * self.action_std + self.action_mean
