from typing import Optional

import torch
from torch.nn.modules.transformer import _get_clones
from torch import nn, Tensor


class ChunkEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        max_chunk_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pos_emb = nn.Parameter(torch.randn(1, max_chunk_len + 1, hidden_dim) * 0.02)
        self.chunk_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, de_action_head, chunks: torch.Tensor) -> torch.Tensor:
        """
        chunks: [B, S, T, D]
        return: [B, S, H]
        """
        b, s, t, d = chunks.shape

        x = chunks.reshape(b * s, t, d)
        x = de_action_head(x)  # [B*S, T, H]

        token = self.chunk_token.expand(b * s, 1, -1)
        x = torch.cat([token, x], dim=1)  # [B*S, T+1, H]

        x = x + self.pos_emb[:, : t + 1]
        x = self.encoder(x)

        chunk_feat = self.norm(x[:, 0])  # [B*S, H]
        return chunk_feat.reshape(b, s, -1)