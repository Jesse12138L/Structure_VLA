"""The regression head the baseline uses: prefix in, a whole action chunk out, one shot.

One learnable query attends over the prefix, giving a single vector that stands for the
observation and the instruction together, and an MLP turns that vector into every action
of the chunk at once. The state is added to the query before the attention, so the summary
can account for it.

There is no noise and no iterative sampling here. That is the point of having it: run it
against the flow matching expert with everything else held fixed, and the difference is
what the flow matching machinery buys.
"""

import torch
import torch.nn as nn


class RegressionActionExpert(nn.Module):
    """Predicts a chunk of actions directly, without denoising."""

    def __init__(self, prefix_size, action_dim, state_dim, chunk_size,
                 d_head=512, n_heads=8, d_ff=2048, dropout=0.0):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.prefix_proj = nn.Linear(prefix_size, d_head)
        self.state_in = nn.Linear(state_dim, d_head)
        self.query = nn.Parameter(torch.zeros(1, 1, d_head))
        self.attention = nn.MultiheadAttention(d_head, n_heads, dropout=dropout,
                                               batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(d_head, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, chunk_size * action_dim),
        )

    def forward(self, prefix, prefix_mask, state):
        """prefix: (batch, L, prefix_size). prefix_mask: (batch, L) True on real tokens.
        state: (batch, state_dim).

        Returns the predicted actions, (batch, chunk, action_dim).
        """
        memory = self.prefix_proj(prefix)                      # (batch, L, d_head)
        query = self.query + self.state_in(state)[:, None, :]  # (batch, 1, d_head)
        # MultiheadAttention's key padding mask also uses True to mean "ignore"
        summary, _ = self.attention(query, memory, memory, key_padding_mask=~prefix_mask)
        # attention keeps the query's length, and there is one query, so (batch, 1, d_head)
        predicted = self.head(summary[:, 0, :]) # 取那唯一的第 0 个位置             # (batch, chunk * action_dim)
        return predicted.reshape(-1, self.chunk_size, self.action_dim)
