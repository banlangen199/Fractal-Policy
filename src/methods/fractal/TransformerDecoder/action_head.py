from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Leaf action generator modeled after PixelLoss style APIs.

    This module autoregressively predicts the final action dimensions from a
    conditioning vector and encoder memory.
    """

    def __init__(
        self,
        c_channels: int,     
        width: int,
        depth: int,
        num_heads: int,
        action_dim: int = 8,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
        loss_type: str = "l1",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = c_channels

        # Leaf stage uses direct regression from condition to action.
        del depth, num_heads, dim_feedforward, dropout, loss_type, width
        self.cond_to_action = nn.Linear(c_channels, action_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.cond_to_action.weight)
        if self.cond_to_action.bias is not None:
            nn.init.constant_(self.cond_to_action.bias, 0)


    def forward(
        self,
        actions: torch.Tensor,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        cond_list: Any,
    ) -> torch.Tensor:
        pred = self.cond_to_action(cond_list[0])
        return pred.view(actions.shape[0], actions.shape[1], self.action_dim)
    
    def sample(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        next_level_sample_fn: Optional[Any] = None,
        cond_list: Any = None,
        num_iter: Optional[int] = None,
        cfg: float = 1.0,
        cfg_schedule: str = "constant",
        temperature: float = 1.0,
        filter_threshold: float = 0.0,
    ) -> torch.Tensor:
        # For sampling, we just do a single forward pass to get the action.
        return self.cond_to_action(cond_list[0])


