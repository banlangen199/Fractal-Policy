from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        x_pos: Optional[torch.Tensor] = None,
        mem_pos: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        q = x if x_pos is None else x + x_pos
        attn_mask = None
        if causal:
            l = q.shape[1]
            attn_mask = torch.triu(
                torch.ones(l, l, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
        self_attn_out = self.self_attn(
            q,
            q,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        x = self.norm1(x + self.dropout1(self_attn_out))

        q = x if x_pos is None else x + x_pos
        k = memory if mem_pos is None else memory + mem_pos
        cross_attn_out = self.cross_attn(q, k, memory, need_weights=False)[0]
        x = self.norm2(x + self.dropout2(cross_attn_out))

        x = self.norm3(x + self.ffn(x))
        return x


class MARActionGenerator(nn.Module):
    def __init__(
        self,
        seq_len: int,
        sub_trunk_size: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        cond_dim: int,
        dropout: float,
        action_dim: int = 8,
        num_iters: int = 4,
        num_conds: int = 3,
        latent_loss_type: str= "l1",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sub_trunk_size = sub_trunk_size
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.num_iters = num_iters
        self.num_conds = num_conds
        self.latent_loss_type = latent_loss_type

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.chunk_flat_proj = nn.Linear(sub_trunk_size * action_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + num_conds, hidden_dim) * 0.02)

        dim_feedforward = hidden_dim * 4
        self.blocks = nn.ModuleList(
            [
                SelfCrossAttentionBlock(
                    hidden_dim=hidden_dim,
                    nheads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _run_blocks(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x_pos = self.pos_embed[:, : x.shape[1], :]
        for blk in self.blocks:
            x = blk(
                x,
                memory=memory,
                x_pos=x_pos,
                mem_pos=mem_pos,
                causal=False,
            )
        return self.norm(x)

    def _align_memory_batch(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        target_batch: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        cur_batch = memory.shape[0]
        if cur_batch == target_batch:
            return memory, mem_pos
        if target_batch % cur_batch != 0:
            raise ValueError(
                f"Cannot align memory batch {cur_batch} to target {target_batch}."
            )
        repeat = target_batch // cur_batch
        memory = memory.unsqueeze(1).expand(cur_batch, repeat, memory.shape[1], memory.shape[2]).reshape(
            target_batch, memory.shape[1], memory.shape[2]
        )
        if mem_pos is not None:
            mem_pos = mem_pos.unsqueeze(1).expand(cur_batch, repeat, mem_pos.shape[1], mem_pos.shape[2]).reshape(
                target_batch, mem_pos.shape[1], mem_pos.shape[2]
            )
        return memory, mem_pos

    def _get_conds(self, cond_list: Any, batch_size: int) -> list[torch.Tensor]:
        if isinstance(cond_list, dict):
            conds = cond_list.get("conds", [])
        elif isinstance(cond_list, (list, tuple)):
            conds = list(cond_list)
        elif torch.is_tensor(cond_list):
            conds = [cond_list]
        else:
            raise ValueError("cond_list must be dict/list/tuple/tensor and cannot be empty")

        if len(conds) == 0:
            raise ValueError("cond_list contains no condition tensors")

        for c in conds:
            if c.ndim != 2:
                raise ValueError(f"Each condition must be 2D [B, C], got shape {tuple(c.shape)}")
            if c.shape[0] != batch_size:
                raise ValueError(f"Condition batch mismatch: expected {batch_size}, got {c.shape[0]}")
            if c.shape[1] != self.cond_dim:
                raise ValueError(f"Condition dim mismatch: expected {self.cond_dim}, got {c.shape[1]}")
        return conds

    def _chunk_to_feat(self, chunks: torch.Tensor) -> torch.Tensor:
        # chunks: [B, seq_len, sub_trunk_size, D] -> [B, seq_len, H]
        if chunks.shape[-1] != self.action_dim:
            raise ValueError(
                f"MARActionGenerator expects action_dim={self.action_dim}, got {chunks.shape[-1]}"
            )
        b, s, t, d = chunks.shape
        if t != self.sub_trunk_size:
            raise ValueError(
                f"Chunk length mismatch: expected sub_trunk_size={self.sub_trunk_size}, got {t}"
            )
        flat = chunks.reshape(b * s, t * d)
        feat = self.chunk_flat_proj(flat)
        return feat.reshape(b, s, self.hidden_dim)

    def _build_next_conds(self, middle_cond: torch.Tensor) -> list[torch.Tensor]:
        """
        Build next-level conditions for 1D action sequence.

        middle_cond: [B, seq_len, H]

        Recommended 1D conditions:
            middle: current token condition
            prev:   previous temporal neighbor
            next:   next temporal neighbor
        """
        b, seq_len, hdim = middle_cond.shape

        middle = middle_cond

        zeros_left = torch.zeros(
            b, 1, hdim,
            device=middle_cond.device,
            dtype=middle_cond.dtype,
        )
        zeros_right = torch.zeros(
            b, 1, hdim,
            device=middle_cond.device,
            dtype=middle_cond.dtype,
        )

        prev_cond = torch.cat(
            [zeros_left, middle_cond[:, :-1, :]],
            dim=1,
        )

        next_cond = torch.cat(
            [middle_cond[:, 1:, :], zeros_right],
            dim=1,
        )

        if self.num_conds == 1:
            return [middle]

        if self.num_conds == 3:
            return [middle, prev_cond, next_cond]

        if self.num_conds == 4:
            # Optional global anchor.
            # For REVERSE target, position 0 can be treated as keyframe-side anchor.
            global_cond = middle_cond[:, :1, :].expand(-1, seq_len, -1)
            return [middle, prev_cond, next_cond, global_cond]

        raise ValueError(
            f"Unsupported num_conds={self.num_conds} for 1D MAR. "
            "Use 1, 3, or 4."
        )

    def _random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # True means masked.
        mask_rates = torch.empty(batch_size, device=device).uniform_(0.15, 0.85)
        num_masked = (self.seq_len * mask_rates).round().long()
        num_masked = torch.clamp(num_masked, min=1, max=max(1, self.seq_len - 1))

        rand_order = torch.argsort(torch.rand(batch_size, self.seq_len, device=device), dim=1)
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            mask[i, rand_order[i, : num_masked[i]]] = True
        return mask

    def _build_reveal_mask(
        self,
        known_mask: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        if step == total_steps - 1:
            return ~known_mask

        remaining = (~known_mask).sum(dim=1)
        ratio = math.cos(math.pi / 2.0 * float(step + 1) / float(total_steps))
        next_mask_num = torch.floor(remaining.float() * ratio).long()
        max_next_mask_num = remaining - 1
        next_mask_num = torch.where(
            remaining > 1,
            torch.minimum(torch.clamp(next_mask_num, min=1), max_next_mask_num),
            torch.zeros_like(next_mask_num),
        )
        num_to_reveal = remaining - next_mask_num

        reveal_mask = torch.zeros_like(known_mask)
        for i in range(known_mask.shape[0]):
            idx = torch.nonzero(~known_mask[i], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            perm = idx[torch.randperm(idx.numel(), device=idx.device)]
            k = int(num_to_reveal[i].item())
            if k > 0:
                reveal_mask[i, perm[:k]] = True
        return reveal_mask

    def predict(
        self,
        sub_chunks: torch.Tensor,
        cond_list: Any,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        if sub_chunks.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {sub_chunks.shape[1]}")

        b = sub_chunks.shape[0]
        memory, mem_pos = self._align_memory_batch(memory, mem_pos, b)
        conds = self._get_conds(cond_list, batch_size=b)
        cond_count = min(len(conds), self.num_conds)

        chunk_feats = self._chunk_to_feat(sub_chunks)
        mask = self._random_mask(b, chunk_feats.device)
        masked_tokens = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.expand_as(chunk_feats),
            chunk_feats,
        )

        prefix = [self.cond_proj(conds[i]).unsqueeze(1) for i in range(cond_count)]
        x = torch.cat(prefix + [masked_tokens], dim=1)
        h = self._run_blocks(x, memory=memory, mem_pos=mem_pos)
        middle_cond = h[:, cond_count:, :]
        return self._build_next_conds(middle_cond)

    def forward(
        self,
        actions: torch.Tensor,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        cond_list: Any = None,
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        # actions: [B, current_len, D] -> sub_chunks: [B, seq_len, sub_trunk_size, D]
        b, l, d = actions.shape
        if l % self.seq_len != 0:
            raise ValueError(f"current_len={l} is not divisible by seq_len={self.seq_len}")
        sub_trunk_size = l // self.seq_len
        if sub_trunk_size != self.sub_trunk_size:
            raise ValueError(
                f"Expected sub_trunk_size={self.sub_trunk_size}, got {sub_trunk_size}"
            )
        sub_chunks = actions.view(b, self.seq_len, sub_trunk_size, d)

        memory, mem_pos = self._align_memory_batch(memory, mem_pos, b)
        conds = self._get_conds(cond_list, batch_size=b)

        cond_list_next = self.predict(
            sub_chunks=sub_chunks,
            cond_list=conds,
            memory=memory,
            mem_pos=mem_pos,
        )

        #latent_loss
        sub_chunks_feat = self._chunk_to_feat(sub_chunks)
        target=sub_chunks_feat.detach()

        loss_type=self.latent_loss_type
        
        if loss_type=='mse':
            latent_loss=F.mse_loss(cond_list_next[0], target, reduction='none')
        elif loss_type=='cosine':
            latent_loss=1 - F.cosine_similarity(cond_list_next[0], target, dim=-1)
        elif loss_type=='l1':
            latent_loss=F.l1_loss(cond_list_next[0], target, reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type={loss_type}")


        actions = sub_chunks.reshape(b * self.seq_len, sub_trunk_size, d)
        cond_list_next = [c.reshape(b * self.seq_len, -1) for c in cond_list_next]

        return actions, cond_list_next, latent_loss

    def sample(
        self,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        next_level_sample_fn: Callable[..., Any],
        cond_list: Any = None,
        num_iter: Optional[int] = None,
        cfg: float = 1.0,
        cfg_schedule: str = "constant",
        temperature: float = 1.0,
        filter_threshold: float = 0.0,
    ) -> torch.Tensor:
        memory, mem_pos = self._align_memory_batch(memory, mem_pos, memory.shape[0])
        conds = self._get_conds(cond_list, batch_size=memory.shape[0])
        b = conds[0].shape[0]
        cond_count = min(len(conds), self.num_conds)
        cond_tokens = [self.cond_proj(conds[i]).unsqueeze(1) for i in range(cond_count)]

        num_steps = self.num_iters if num_iter is None else max(1, int(num_iter))
        num_steps = min(self.seq_len, num_steps)

        tokens = self.mask_token.expand(b, self.seq_len, -1).clone()
        known_mask = torch.zeros(b, self.seq_len, dtype=torch.bool, device=memory.device)
        sampled_chunks = [None for _ in range(self.seq_len)]

        for step in range(num_steps):
            x = torch.cat(cond_tokens + [tokens], dim=1)
            h = self._run_blocks(x, memory=memory, mem_pos=mem_pos)
            cond_all = h[:, cond_count:, :]

            reveal_mask = self._build_reveal_mask(known_mask, step, num_steps)
            if cfg_schedule == "linear":
                cfg_iter = 1.0 + (cfg - 1.0) * float(step + 1) / float(self.seq_len)
            else:
                cfg_iter = cfg

            for pos in range(self.seq_len):
                reveal_pos = reveal_mask[:, pos]
                if not reveal_pos.any():
                    continue

                cond_pos = cond_all[reveal_pos, pos, :]
                cond_payload = [cond_pos]
                memory_pos = memory[reveal_pos]
                mem_pos_pos = mem_pos[reveal_pos] if mem_pos is not None else None

                sampled_out = next_level_sample_fn(
                    memory=memory_pos,
                    mem_pos=mem_pos_pos,
                    cond_list=cond_payload,
                    cfg=cfg_iter,
                    temperature=temperature,
                    filter_threshold=filter_threshold,
                )
                sampled = sampled_out[0] if isinstance(sampled_out, tuple) else sampled_out
                if sampled.ndim == 2:
                    sampled = sampled.unsqueeze(1)

                feat = self._chunk_to_feat(sampled.unsqueeze(1)).squeeze(1)
                tokens[reveal_pos, pos, :] = feat

                idxs = torch.nonzero(reveal_pos, as_tuple=False).squeeze(-1)
                for j, bidx in enumerate(idxs.tolist()):
                    if sampled_chunks[pos] is None:
                        sampled_chunks[pos] = torch.zeros(
                            b,
                            sampled.shape[1],
                            sampled.shape[2],
                            device=sampled.device,
                            dtype=sampled.dtype,
                        )
                    sampled_chunks[pos][bidx] = sampled[j]

            known_mask = known_mask | reveal_mask

        out_chunks = []
        x = torch.cat(cond_tokens + [tokens], dim=1)
        h = self._run_blocks(x, memory=memory, mem_pos=mem_pos)
        cond_all = h[:, cond_count:, :]
        for pos in range(self.seq_len):
            if sampled_chunks[pos] is None:
                cond_payload = [cond_all[:, pos, :]]
                sampled_out = next_level_sample_fn(
                    memory=memory,
                    mem_pos=mem_pos,
                    cond_list=cond_payload,
                    cfg=cfg,
                    temperature=temperature,
                    filter_threshold=filter_threshold,
                )
                sampled = sampled_out[0] if isinstance(sampled_out, tuple) else sampled_out
                if sampled.ndim == 2:
                    sampled = sampled.unsqueeze(1)
                sampled_chunks[pos] = sampled
            out_chunks.append(sampled_chunks[pos])

        chunks = torch.stack(out_chunks, dim=1)  # [B, seq_len, next_len, D]
        pred_actions = chunks.flatten(1, 2)
        return pred_actions
