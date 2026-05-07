from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache(nn.Module):
    def __init__(self, max_batch_size: int, max_seq_length: int, n_head: int, head_dim: int):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape), persistent=False)

    def update(self, input_pos: int, k_val: torch.Tensor, v_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # k_val/v_val: [B, H, S, D]
        bsz, _, step_len, _ = k_val.shape
        self.k_cache[:bsz, :, input_pos : input_pos + step_len, :] = k_val.to(self.k_cache.dtype)
        self.v_cache[:bsz, :, input_pos : input_pos + step_len, :] = v_val.to(self.v_cache.dtype)
        return (
            self.k_cache[:bsz, :, : input_pos + step_len, :],
            self.v_cache[:bsz, :, : input_pos + step_len, :],
        )


class SelfAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dropout: float):
        super().__init__()
        if hidden_dim % nheads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by nheads={nheads}")

        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.head_dim = hidden_dim // nheads
        self.attn_dropout = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.kv_cache: Optional[KVCache] = None

    def setup_kv_cache(self, max_batch_size: int, max_seq_length: int, device: torch.device, dtype: torch.dtype) -> None:
        self.kv_cache = KVCache(max_batch_size, max_seq_length, self.nheads, self.head_dim).to(
            device=device,
            dtype=dtype,
        )

    def clear_kv_cache(self) -> None:
        self.kv_cache = None

    def _project_qkv(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, step_len, _ = q_in.shape

        q = self.q_proj(q_in).view(bsz, step_len, self.nheads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(bsz, step_len, self.nheads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).view(bsz, step_len, self.nheads, self.head_dim).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        x_pos: Optional[torch.Tensor] = None,
        causal: bool = False,
        input_pos: Optional[int] = None,
    ) -> torch.Tensor:
        qk_in = x if x_pos is None else x + x_pos
        v_in = x

        q, k, v = self._project_qkv(qk_in, qk_in, v_in)

        if input_pos is not None:
            if self.kv_cache is None:
                raise RuntimeError("KV cache is not initialized. Call setup_kv_cache before sampling.")

            k_all, v_all = self.kv_cache.update(int(input_pos), k, v)
            out = F.scaled_dot_product_attention(
                q,
                k_all,
                v_all,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=causal,
            )

        bsz, _, step_len, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(bsz, step_len, self.hidden_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, nheads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = SelfAttentionWithKVCache(
            hidden_dim=hidden_dim,
            nheads=nheads,
            dropout=dropout,
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

    def setup_kv_cache(self, max_batch_size: int, max_seq_length: int, device: torch.device, dtype: torch.dtype) -> None:
        self.self_attn.setup_kv_cache(
            max_batch_size=max_batch_size,
            max_seq_length=max_seq_length,
            device=device,
            dtype=dtype,
        )

    def clear_kv_cache(self) -> None:
        self.self_attn.clear_kv_cache()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        x_pos: Optional[torch.Tensor] = None,
        mem_pos: Optional[torch.Tensor] = None,
        causal: bool = False,
        input_pos: Optional[int] = None,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(
            x=x,
            x_pos=x_pos,
            causal=causal,
            input_pos=input_pos,
        )

        x = self.norm1(x + self.dropout1(self_attn_out))

        q = x if x_pos is None else x + x_pos
        k = memory if mem_pos is None else memory + mem_pos
        cross_attn_out = self.cross_attn(q, k, memory, need_weights=False)[0]
        x = self.norm2(x + self.dropout2(cross_attn_out))

        x = self.norm3(x + self.ffn(x))
        return x


class ARActionGenerator(nn.Module):
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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.sub_trunk_size = sub_trunk_size
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.cond_dim = cond_dim

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.chunk_flat_proj = nn.Linear(sub_trunk_size * action_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, hidden_dim) * 0.02)
        dim_feedforward = hidden_dim * 4
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
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
        input_pos: Optional[int] = None,
    ) -> torch.Tensor:
        if input_pos is None:
            x_pos = self.pos_embed[:, : x.shape[1], :]
        else:
            x_pos = self.pos_embed[:, input_pos : input_pos + x.shape[1], :]

        for blk in self.blocks:
            x = blk(
                x,
                memory=memory,
                x_pos=x_pos,
                mem_pos=mem_pos,
                causal=True,
                input_pos=input_pos,
            )
        return self.norm(x)

    def _setup_kv_cache(self, max_batch_size: int, max_seq_length: int, device: torch.device, dtype: torch.dtype) -> None:
        for blk in self.blocks:
            blk.setup_kv_cache(
                max_batch_size=max_batch_size,
                max_seq_length=max_seq_length,
                device=device,
                dtype=dtype,
            )

    def _clear_kv_cache(self) -> None:
        for blk in self.blocks:
            blk.clear_kv_cache()

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
                f"ARActionGenerator expects action_dim={self.action_dim}, got {chunks.shape[-1]}"
            )
        b, s, t, d = chunks.shape
        if t != self.sub_trunk_size:
            raise ValueError(
                f"Chunk length mismatch: expected sub_trunk_size={self.sub_trunk_size}, got {t}"
            )
        flat = chunks.reshape(b * s, t * d)
        feat = self.chunk_flat_proj(flat)
        return feat.reshape(b, s, self.hidden_dim)

    def predict(
        self,
        sub_chunks: torch.Tensor,
        cond_list: Any,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        input_pos: Optional[int] = None,
    ) -> list[torch.Tensor]:
        if sub_chunks.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {sub_chunks.shape[1]}")

        # memory batch alignment and cond validation are done by caller (forward/sample)
        conds = cond_list
        cond_token = self.cond_proj(conds[0]).unsqueeze(1)
        sub_chunks_feat = self._chunk_to_feat(sub_chunks)
        teacher_in = torch.cat([cond_token, sub_chunks_feat], dim=1)

        if input_pos is not None:
            # Keep a single-step sequence dimension for KV-cache decoding.
            teacher_in = teacher_in[:, input_pos : input_pos + 1, :]
        
        h = self._run_blocks(
            teacher_in,
            memory=memory,
            mem_pos=mem_pos,
            input_pos=input_pos,
        )

        if input_pos is not None:
            middle_cond = h[:, 0, :]
        else:
            middle_cond = h[:, :-1, :]

        return [middle_cond]

    def forward(
        self,
        actions: torch.Tensor,
        memory: torch.Tensor,
        mem_pos: Optional[torch.Tensor],
        cond_list: Any = None,
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """Training"""
        # actions: [B, S, D] -> sub_chunks: [B, seq_len, sub_trunk_size, D]
        b, s, d = actions.shape
        if s % self.seq_len != 0:
            raise ValueError(f"current_len={s} is not divisible by seq_len={self.seq_len}")
        sub_chunk_size = s // self.seq_len
        if sub_chunk_size != self.sub_trunk_size:
            raise ValueError(
                f"Expected sub_chunk_size={self.sub_trunk_size}, got {sub_chunk_size}"
            )
        sub_chunks = actions.view(b, self.seq_len, sub_chunk_size, d)

        memory, mem_pos = self._align_memory_batch(memory, mem_pos, b)
        conds = self._get_conds(cond_list, batch_size=b)

        # Get next-level conditions from current-level sub-chunks.
        cond_list_next = self.predict(
            sub_chunks=sub_chunks,
            cond_list=conds,
            memory=memory,
            mem_pos=mem_pos,
        )

        #latent_loss
        target = self._chunk_to_feat(sub_chunks)

        # target=sub_chunks_feat.detach()
        #TODO: make loss type configurable
        loss_type='l1'
        if loss_type=='mse':
            latent_loss=F.mse_loss(cond_list_next[0], target, reduction='none')
        elif loss_type=='cosine':
            latent_loss=1 - F.cosine_similarity(cond_list_next[0], target, dim=-1)
        elif loss_type=='l1':
            latent_loss=F.l1_loss(cond_list_next[0], target, reduction='none')
        else:
            raise ValueError(f"Unsupported loss_type={loss_type}")


        # Pass all sub-chunks recursively: [B, seq_len, sub_trunk_size, D] -> [B*seq_len, sub_trunk_size, D]
        actions = sub_chunks.reshape(b * self.seq_len, sub_chunk_size, d)
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
    ) :
        """inference"""
        memory, mem_pos = self._align_memory_batch(memory, mem_pos, memory.shape[0])
        conds = self._get_conds(cond_list, batch_size=memory.shape[0])
        b = conds[0].shape[0]
        sub_chunks = torch.zeros(
            b,
            self.seq_len,
            self.sub_trunk_size,
            self.action_dim,
            device=memory.device,
            dtype=memory.dtype,
        )

        num_steps = self.seq_len 
        self._setup_kv_cache(max_batch_size=b, max_seq_length=num_steps, device=memory.device, dtype=memory.dtype)
        try:
            for step in range(num_steps):
                cur_chunks=sub_chunks.clone() 


                cond_list_next = self.predict(
                    sub_chunks=cur_chunks,
                    cond_list=conds,
                    memory=memory,
                    mem_pos=mem_pos,
                    input_pos=step,
                )

                if cfg_schedule == "linear":
                    cfg_iter = 1.0 + (cfg - 1.0) * float(step + 1) / float(self.seq_len)
                else:
                    cfg_iter = cfg

                sampled_out = next_level_sample_fn(
                    memory=memory,
                    mem_pos=mem_pos,
                    cond_list=cond_list_next,
                    cfg=cfg_iter,
                    temperature=temperature,
                    filter_threshold=filter_threshold,
                )
                sampled_chunk = sampled_out[0] if isinstance(sampled_out, tuple) else sampled_out
                if sampled_chunk.ndim == 2:
                    sampled_chunk = sampled_chunk.unsqueeze(1)

                cur_chunks[:, step, :, :] = sampled_chunk.to(cur_chunks.dtype)
                sub_chunks=cur_chunks.clone()
        finally:
            self._clear_kv_cache()

        pred_actions = sub_chunks.flatten(1, 2)

        return pred_actions


if __name__ == "__main__":
    # Quick test to verify the model can run without errors
    batch_size = 2
    seq_len = 4
    sub_trunk_size = 2
    action_dim = 8
    hidden_dim = 512
    num_heads = 4
    num_blocks = 2
    cond_dim = 512
    dropout = 0.1

    model = ARActionGenerator(
        seq_len=seq_len,
        sub_trunk_size=sub_trunk_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        cond_dim=cond_dim,
        dropout=dropout,
        action_dim=action_dim,
    )

    actions = torch.randn(batch_size, seq_len * sub_trunk_size, action_dim)
    memory = torch.randn(batch_size, 10, hidden_dim)
    mem_pos = torch.randn(1, 10, hidden_dim)
    cond_list = [torch.randn(batch_size, cond_dim)]

    pred_actions, cond_list_next, aux_loss = model(
        actions=actions,
        memory=memory,
        mem_pos=mem_pos,
        cond_list=cond_list,
    )
    print("Predicted actions shape:", pred_actions.shape)


