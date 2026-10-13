"""The flow matching action expert: a small transformer decoder that reads the prefix.

The prefix is the frozen trunk's output and never changes. The suffix is what this module
produces: one token for the robot state, then one token per action in the chunk. A suffix
token carries both the noisy action and the noise level it sits at, so the time is
embedded and added to the action embedding.

Two places where this differs from SmolVLA, both deliberate:

* The cross-attention keys and values come from the prefix through this module's own
  Linear, not from the trunk's internal key/value tensors. SmolVLA reshapes its expert
  projections so they accept the trunk's key/value width and reads those directly.
  Projecting the hidden states here is the same idea, needs no width matching, and is
  much shorter.
* The state is a suffix token, where SmolVLA puts it in the prefix. In the prefix, the
  state would go through a trainable projection before entering the frozen trunk, so
  gradients would have to flow back through all of it: measured about 9x slower than
  letting the trunk run without gradients.

The padding masks are not optional. The action chunk is padded near the end of an episode,
and the padded values are the episode's last action repeated - plausible-looking but
wrong - so valid positions must not attend to them. Skipping the mask does not raise, it
just quietly trains a worse model.
"""

import torch
import torch.nn as nn

from ..common.flow_matching import sinusoidal_time_embedding


class FlowActionExpert(nn.Module):
    """Predicts the flow-matching velocity for a chunk of noisy actions."""

    def __init__(self, prefix_size, action_dim, state_dim, chunk_size,
                 d_exp=512, n_heads=8, n_layers=4, d_ff=2048, dropout=0.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_exp = d_exp

        self.prefix_proj = nn.Linear(prefix_size, d_exp)
        self.action_in = nn.Linear(action_dim, d_exp)
        self.state_in = nn.Linear(state_dim, d_exp)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_exp, d_exp),
            nn.SiLU(),
            nn.Linear(d_exp, d_exp),
        )

        layer = nn.TransformerDecoderLayer(d_exp, n_heads, d_ff, dropout=dropout,
                                           batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.action_out = nn.Linear(d_exp, action_dim)

    def forward(self, prefix, prefix_mask, noisy_action, time, state, action_is_pad):
        """prefix: (batch, L, prefix_size). prefix_mask: (batch, L) True on real tokens.
        noisy_action: (batch, chunk, action_dim). time: (batch,).
        state: (batch, state_dim). action_is_pad: (batch, chunk) True on padded steps.

        Returns the predicted velocity, (batch, chunk, action_dim).
        """
        batch = prefix.shape[0]

        action_tokens = self.action_in(noisy_action)  # (batch, chunk, d_exp)
        time_tokens = self.time_mlp(sinusoidal_time_embedding(time, self.d_exp))  # (batch, d_exp)
        action_tokens = action_tokens + time_tokens[:, None, :]  # (batch, chunk, d_exp)

        state_token = self.state_in(state)[:, None, :]  # (batch, 1, d_exp)
        suffix = torch.cat([state_token, action_tokens], dim=1)  # (batch, 1 + chunk, d_exp)

        # The state token is never padding. `tgt_key_padding_mask` follows PyTorch's
        # convention throughout: True means "ignore this position".
        state_is_pad = torch.zeros(batch, 1, dtype=torch.bool, device=prefix.device)
        suffix_padding_mask = torch.cat([state_is_pad, action_is_pad], dim=1)  # (batch, 1 + chunk)

        memory = self.prefix_proj(prefix)  # (batch, L, d_exp)
        decoded = self.decoder(
            suffix,
            memory,
            tgt_key_padding_mask=suffix_padding_mask,
            memory_key_padding_mask=~prefix_mask,
        )  # (batch, 1 + chunk, d_exp)
        # drop the state token's output; only the action tokens carry a prediction
        return self.action_out(decoded[:, 1:, :])  # (batch, chunk, action_dim)
