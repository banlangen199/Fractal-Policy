from __future__ import annotations

from typing import Optional, Tuple
from functools import partial

import torch
import torch.nn as nn

from .action_head import ActionHead
from .ar_action import ARActionGenerator
from .mar_action import MARActionGenerator


class FractalAction(nn.Module):
    def __init__(
        self,
        action_size_list,
        hidden_dim_list,
        num_blocks_list,
        num_heads_list,
        generator_type_list,
        dropout: float = 0.1,
        action_dim: int = 8,
        use_lang_cond: bool = False,
        fractal_level: int = 0,
        latent_loss_type: str = "l1",
        memory_access_list: Optional[list[bool]] = None,
        memory_dim: Optional[int] = 512,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.action_dim = action_dim
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(hidden_dim_list)
        self.current_len = action_size_list[fractal_level]
        self.hidden_dim = hidden_dim_list[fractal_level]

        if self.fractal_level == 0:
            # Start token for first-level autoregressive decoding 
            self.sos_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

        if generator_type_list[fractal_level] == "ar":
            generator = ARActionGenerator
        elif generator_type_list[fractal_level] == "mar":
            generator = MARActionGenerator
        else:
            raise NotImplementedError

        self.generator = generator(
            seq_len=(action_size_list[self.fractal_level] // action_size_list[self.fractal_level+1]),
            sub_trunk_size=action_size_list[self.fractal_level+1],
            hidden_dim=hidden_dim_list[self.fractal_level],
            num_heads=num_heads_list[self.fractal_level],
            num_blocks=num_blocks_list[self.fractal_level],
            action_dim=action_dim,
            cond_dim=hidden_dim_list[self.fractal_level-1] if self.fractal_level > 0 else hidden_dim_list[0],
            dropout=dropout,
            latent_loss_type=latent_loss_type,
            use_memory=memory_access_list[self.fractal_level] if memory_access_list is not None else True,
            memory_dim=memory_dim,
        )

        # Recursive next level, same pattern as fractalgen FractalGen.
        if self.fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalAction(
                action_size_list=action_size_list,
                hidden_dim_list=hidden_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list, 
                generator_type_list=generator_type_list,
                dropout=dropout,
                action_dim=action_dim,
                use_lang_cond=use_lang_cond,
                fractal_level=self.fractal_level + 1,
                latent_loss_type=latent_loss_type,
                memory_access_list=memory_access_list,
                memory_dim=memory_dim,
            )
        else:
            self.next_fractal = ActionHead(
                c_channels=hidden_dim_list[self.fractal_level],
                action_dim=action_dim,  
            )


    def forward(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        actions: Optional[torch.Tensor],
        proprio,
        is_pad,
        action_head,
        de_action_head,
        cond_list: torch.Tensor = None,
    ):
        """
        Forward pass to get loss recursively.
        """
        if actions is None:
            raise ValueError("Training forward requires ground-truth actions.")
        if actions.shape[1] != self.current_len:
            raise ValueError(
                f"Level {self.fractal_level} expects action len={self.current_len}, got {actions.shape[1]}"
            )
        
        batch_size = actions.shape[0]

        if self.fractal_level == 0:
            root_cond = self.sos_embedding[:, 0, :].expand(batch_size, -1)
            cond_list = [root_cond for _ in range(3)]

        actions, next_cond_list, latent_loss=self.generator(
            actions=actions,
            cond_list=cond_list,
            memory=memory,
            mem_pos=mem_pos,
            proprio=proprio,
            is_pad=is_pad,
            action_head=action_head,
            de_action_head=de_action_head,
        )

        pred_actions,child_loss=self.next_fractal(
            actions=actions, 
            cond_list=next_cond_list,                    
            memory=memory, 
            mem_pos=mem_pos,
            proprio=proprio,
            is_pad=is_pad,
            action_head=action_head,
            de_action_head=de_action_head,
        )
        

        if pred_actions.ndim == 2:
            pred_actions = pred_actions.unsqueeze(1)
        
        latent_losses = [latent_loss] + child_loss
        return pred_actions.reshape(batch_size, self.current_len, self.action_dim), latent_losses

    
    def sample(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        proprio,
        action_head,
        de_action_head,
        cond_list: torch.Tensor = None,
        num_iter_list: Optional[list[int]] = None,
    ):
        """
        Generate samples recursively.
        """

        # if num_iter_list is None:
        #     num_iter_list = [64,16,1]

        if self.fractal_level < len(num_iter_list):
            num_iter = int(num_iter_list[self.fractal_level])
        else:
            num_iter = None

        if self.fractal_level == 0:
            batch_size = memory.shape[0]
            root_cond = self.sos_embedding[:, 0, :].expand(batch_size, -1)
            cond_list = [root_cond for _ in range(3)]

        if self.fractal_level < self.num_fractal_levels - 2:
            next_level_sample_function = partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
            )
        else:
            next_level_sample_function = self.next_fractal.sample

        return self.generator.sample(
            cond_list=cond_list,
            memory=memory,
            mem_pos=mem_pos,
            proprio=proprio,
            action_head=action_head,
            de_action_head=de_action_head,
            num_iter=num_iter,
            next_level_sample_fn=next_level_sample_function,
        )
    
    def levelwise_sample(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        proprio,
        action_head,
        de_action_head,
        cond_list: torch.Tensor = None,
        num_iter_list: Optional[list[int]] = None,

    ):
        """
        Level-wise sampling.

        Different from sample():

        sample():
            depth-first recursive generation.

        levelwise_sample():
            current level first generates all next-level conditions,
            then passes the whole condition sequence to the next level.
        """
        del num_iter_list

        if self.fractal_level == 0:
            batch_size = memory.shape[0]
            root_cond = self.sos_embedding[:, 0, :].expand(batch_size, -1)
            cond_list = [root_cond for _ in range(3)]

        if not hasattr(self.generator, "sample_cond_sequence"):
            raise NotImplementedError(
                f"levelwise_sample currently requires ARActionGenerator, "
                f"but got {type(self.generator)}"
            )

        # 1. Current level generates all next-level condition tokens first.
        # next_cond_seq: [B, seq_len, hidden_dim]
        next_cond_seq = self.generator.sample_cond_sequence(
            memory=memory,
            mem_pos=mem_pos,
            proprio=proprio,
            action_head=action_head,
            de_action_head=de_action_head,
            cond_list=cond_list,
        )

        batch_size, seq_len, hidden_dim = next_cond_seq.shape

        # 2. Flatten [B, seq_len, H] -> [B * seq_len, H],
        # matching the training forward path.
        next_cond_flat = next_cond_seq.reshape(batch_size * seq_len, hidden_dim)
        next_cond_list = [next_cond_flat]

        # 3. Send the whole generated condition batch to the next level.
        if isinstance(self.next_fractal, FractalAction):
            child_actions = self.next_fractal.levelwise_sample(
                memory=memory,
                mem_pos=mem_pos,
                proprio=proprio,
                action_head=action_head,
                de_action_head=de_action_head,
                cond_list=next_cond_list,
                num_iter_list=None,
            )
        else:
            # Leaf ActionHead.
            child_actions = self.next_fractal.sample(
                memory=memory,
                mem_pos=mem_pos,
                proprio=proprio,
                action_head=action_head,
                de_action_head=de_action_head,
                cond_list=next_cond_list,
            )

        if child_actions.ndim == 2:
            child_actions = child_actions.unsqueeze(1)

        # child_actions shape:
        #   [B * seq_len, sub_trunk_size, action_dim]
        #
        # reshape back to:
        #   [B, seq_len * sub_trunk_size, action_dim]
        pred_actions = child_actions.reshape(
            batch_size,
            seq_len * child_actions.shape[1],
            self.action_dim,
        )

        return pred_actions


if __name__ == "__main__":
    # Quick test to verify the model can run without errors
    batch_size = 2
    seq_len = 64
    action_dim = 8
    hidden_dim = 512



    model = FractalAction(
        hidden_dim_list=[512, 256, 128],
        action_dim=action_dim,
        action_size_list=[64, 16, 1],
        num_blocks_list=[2, 2, 2],
        generator_type_list=["ar","ar", "ar"],
        fractal_level=0,
    )

    memory = torch.randn(batch_size, seq_len, hidden_dim)
    mem_pos = torch.randn(batch_size, seq_len, hidden_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    cond_list = torch.randn(batch_size, hidden_dim)

    pred_actions= model(memory=memory, mem_pos=mem_pos, actions=actions, cond_list=cond_list)
    print("Predicted actions shape:", pred_actions.shape)

    pred_actions_sampled = model.sample(memory=memory, mem_pos=mem_pos, cond_list=cond_list)