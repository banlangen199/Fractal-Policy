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
        action_dim: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = c_channels

        # Leaf stage uses direct regression from condition to action.
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
        proprio,
        is_pad,
        cond_list: Any,
        action_head,
        de_action_head,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        pred = self.cond_to_action(cond_list[0])

        return pred.view(actions.shape[0], actions.shape[1], self.action_dim), []

    def sample(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        action_head,
        de_action_head,
        next_level_sample_fn: Optional[Any] = None,
        cond_list: Any = None,
        num_iter: Optional[int] = None,
    ) -> torch.Tensor:
        # For sampling, we just do a single forward pass to get the action.
        return self.cond_to_action(cond_list[0])


