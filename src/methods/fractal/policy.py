from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from transformers.optimization import get_scheduler

from src.methods.base import BaseMethod, BatchedActionSequence
from src.methods.backbone import build_backbone
from src.methods.fractal.TransformerEncoder import TransformerEncoder, TransformerEncoderLayer
from src.methods.utils import (
    extract_many_from_batch,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim,
        position_embedding,
        lr_backbone,
        masks,
        backbone,
        dilation,
        use_lang_cond,
        use_frozen_bn=False,
    ):
        super().__init__()
        assert len(input_shape) == 4, f"Expected shape (View, C, H, W), but got {input_shape}"
        self._input_shape = tuple(input_shape)

        self.backbone = build_backbone(
            hidden_dim=hidden_dim,
            position_embedding=position_embedding,
            lr_backbone=lr_backbone,
            masks=masks,
            backbone=backbone,
            dilation=dilation,
            use_frozen_bn=use_frozen_bn,
        )
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, task_emb: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._input_shape == x.shape[1:], (
            f"expected input shape {self._input_shape} but got {x.shape[1:]}"
        )

        all_cam_features = []
        all_cam_pos = []
        shape = x.shape
        for cam_id in range(self._input_shape[0]):
            cur_x = x[:, cam_id].reshape(-1, 3, *self._input_shape[2:])
            feat, pos = self.backbone(cur_x)
            feat = self.input_proj(feat[0])
            pos = pos[0]
            all_cam_features.append(feat)
            all_cam_pos.append(pos)

        img_feat = torch.cat(all_cam_features, dim=3)
        img_feat = img_feat.reshape(shape[0], -1, *img_feat.shape[2:])
        pos = torch.cat(all_cam_pos, dim=3)
        return img_feat, pos


class ActorModel(nn.Module):
    def __init__(
        self,
        transformer_decoder,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 3200,
        enc_layers: int = 4,
        pre_norm: bool = True,
        state_dim: int = 8,
        action_dim: int = 8,
        use_lang_cond: bool = False,
        latent_loss_type: str = "l1",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_lang_cond = use_lang_cond

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=pre_norm,
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=enc_layers,
            norm=encoder_norm,
        )

        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.state_mem_pos = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.transformer_decoder = transformer_decoder(
            latent_loss_type=latent_loss_type
            )

    def _build_memory(
        self,
        obs_feat: Tuple[torch.Tensor, torch.Tensor],
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_feat, img_pos = obs_feat
        bs = img_feat.shape[0]

        # [B, C, V, L] -> [V*L, B, C]
        img_tokens = img_feat.flatten(2).permute(2, 0, 1)
        img_pos_tokens = img_pos.flatten(2).permute(2, 0, 1)

        # Position encoding can be shared across batch (B=1). Align it to image batch.
        if img_pos_tokens.shape[1] != bs:
            if img_pos_tokens.shape[1] == 1:
                img_pos_tokens = img_pos_tokens.expand(-1, bs, -1)
            else:
                raise ValueError(
                    f"Position token batch mismatch: pos batch={img_pos_tokens.shape[1]}, expected {bs}."
                )

        if proprio.ndim == 3:
            proprio = proprio[:, 0, :]
        state_token = self.state_proj(proprio).unsqueeze(0)

        src = torch.cat([img_tokens, state_token], dim=0)
        mem_pos = torch.cat([img_pos_tokens, self.state_mem_pos.expand(1, bs, -1)], dim=0)
        encoded = self.transformer_encoder(src, pos=mem_pos)

        # batch-first memory for fractal decoder cross-attention
        memory = encoded.permute(1, 0, 2)
        mem_pos = mem_pos.permute(1, 0, 2)
        return memory, mem_pos

    def forward(
        self,
        obs_feat: Tuple[torch.Tensor, torch.Tensor],
        proprio: torch.Tensor,
        task_embed: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        training: bool = True,
        sample_mode: str = "depth_first",
    ) -> torch.Tensor:
        memory, mem_pos = self._build_memory(obs_feat, proprio)
        if training:
            # print("Training mode: using provided actions for teacher forcing.")
            actions,latent_losses = self.transformer_decoder(
                memory=memory,
                mem_pos=mem_pos,
                actions=actions,
            )
        else: 
            # print("Sampling mode: ignoring provided actions and generating autoregressively.")
            if sample_mode == "depth_first":
                actions = self.transformer_decoder.sample(
                    memory=memory,
                    mem_pos=mem_pos,
                )
            elif sample_mode == "levelwise":
                actions = self.transformer_decoder.levelwise_sample(
                    memory=memory,
                    mem_pos=mem_pos,
                )
            else:
                raise ValueError(
                    f"Unknown sample_mode={sample_mode}. "
                    "Expected 'depth_first' or 'levelwise'."
                )

            latent_losses = None

        return actions, latent_losses



class FractalPolicy(BaseMethod):
    def __init__(
        self,
        encoder_model,
        actor_model,
        lr,
        lr_backbone,
        num_train_steps,
        adaptive_lr,
        weight_decay,
        use_lang_cond,
        action_order,
        action_mode,
        loss_type,
        latent_loss_type,
        actor_grad_clip,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.adaptive_lr = adaptive_lr
        self.weight_decay = weight_decay
        self.num_train_steps = num_train_steps
        self.use_lang_cond = use_lang_cond
        self.action_order = action_order
        self.action_mode = action_mode
        self.loss_type = loss_type
        self.actor_grad_clip = actor_grad_clip

        self.device = self.accelerator.device if self.accelerator else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder_model = encoder_model()
        self.actor_model = actor_model(
            latent_loss_type=latent_loss_type
            )
        self.encoder_model = self.encoder_model.to(self.device)
        self.actor_model = self.actor_model.to(self.device)

        visual_obs_mean = [0.485, 0.456, 0.406]
        visual_obs_std = [0.229, 0.224, 0.225]
        self.img_normalizer = tvf.Normalize(mean=visual_obs_mean, std=visual_obs_std)

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        self.opt = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        if self.adaptive_lr:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.opt,
                num_warmup_steps=100,
                num_training_steps=self.num_train_steps,
            )

        self.prepare_accelerator()

    def training_mode(self, training: bool = True):
        if training:
            self.encoder_model.train()
            self.actor_model.train()
        else:
            self.encoder_model.eval()
            self.actor_model.eval()

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
        training: bool = True,
        sample_mode: str = "depth_first",
    ) -> Union[BatchedActionSequence, Tuple[Any, ...]]:
        raw_img = extract_many_from_batch(batch_input, "rgb")
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw_img, dim=1))
        proprio = batch_input["low_dim_state"]

        if training:
            a_gt = batch_input["action"]
            is_pad = batch_input["is_pad"]
        else:
            a_gt = None
            is_pad = None

        task_emb = batch_input.get("task_emb", None)

        img = self.img_normalizer(img / 255.0)
        obs_feat = self.encoder_model(img, task_emb)
        a_hat, latent_losses = self.actor_model(
            obs_feat,
            proprio,
            task_emb,
            actions=a_gt,
            training=training,
            sample_mode=sample_mode,
        )
        return a_hat, a_gt, is_pad, latent_losses

    @torch.no_grad()
    def act(self, batch_input: dict[str, torch.Tensor],sample_mode:str="depth_first") -> BatchedActionSequence:
        self.training_mode(training=False)
        a_hat, _, _ , _ = self.forward(batch_input, training=False, sample_mode=sample_mode)
        return a_hat

    def validate(
        self,
        batch_input: dict[str, torch.Tensor],
        use_generated_actions: bool = True,
        sample_mode: str = "depth_first",
    ) -> dict[str, torch.Tensor]:
        """
        Offline validation on demonstration data.

        use_generated_actions=True:
            Use the sampling/autoregressive path:
                obs_t -> generated action sequence -> compare with GT

        use_generated_actions=False:
            Use teacher-forcing path:
                obs_t + GT actions -> predicted actions -> compare with GT
        """
        self.training_mode(training=False)

        a_gt = batch_input["action"]
        is_pad = batch_input["is_pad"]

        if use_generated_actions:
            # Real generated sequence validation.
            # This matches evaluation behavior more closely.
            a_hat = self.act(batch_input, sample_mode=sample_mode)
        else:
            # Teacher-forcing validation.
            # This checks supervised prediction quality without autoregressive error accumulation.
            a_hat, a_gt, is_pad, _ = self.forward(
                batch_input,
                training=True,
            )

        if self.loss_type == "l1":
            raw_action_loss = F.l1_loss(a_hat, a_gt, reduction="none")
        elif self.loss_type == "mse":
            raw_action_loss = F.mse_loss(a_hat, a_gt, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # is_pad: [B, T]
        # True  = padding timestep, ignored
        # False = valid timestep
        valid = (~is_pad.bool()).unsqueeze(-1).expand_as(raw_action_loss)
        action_loss = raw_action_loss.masked_fill(~valid, 0.0)

        def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return loss.sum() / mask.sum().clamp_min(1)

        metrics = {
            # Full sequence average loss
            "action_loss": masked_mean(action_loss, valid),

            # Exclude the first action, keeping your previous metric definition
            "traj_loss": masked_mean(action_loss[:, 1:, :], valid[:, 1:, :]),
            "pos_loss": masked_mean(action_loss[:, 1:, :3], valid[:, 1:, :3]),
            "ori_loss": masked_mean(action_loss[:, 1:, 3:7], valid[:, 1:, 3:7]),
            "gripper_loss": masked_mean(action_loss[:, 1:, -1:], valid[:, 1:, -1:]),

            # First action loss
            "first_action_loss": masked_mean(action_loss[:, :1, :], valid[:, :1, :]),
            "first_pos_loss": masked_mean(action_loss[:, :1, :3], valid[:, :1, :3]),
            "first_ori_loss": masked_mean(action_loss[:, :1, 3:7], valid[:, :1, 3:7]),
            "first_gripper_loss": masked_mean(action_loss[:, :1, -1:], valid[:, :1, -1:]),
        }

        # ------------------------------------------------------------
        # Prefix loss:
        # Compare models with different action horizons fairly.
        # For example:
        #   32-16-4-1 model prefix_16_action_loss
        #   vs
        #   16-4-1 model prefix_16_action_loss
        # ------------------------------------------------------------
        prefix_lengths = [1, 2, 4, 8, 16, 32]

        for k in prefix_lengths:
            if k <= action_loss.shape[1]:
                metrics[f"prefix_{k}_action_loss"] = masked_mean(
                    action_loss[:, :k, :],
                    valid[:, :k, :],
                )
                metrics[f"prefix_{k}_pos_loss"] = masked_mean(
                    action_loss[:, :k, :3],
                    valid[:, :k, :3],
                )
                metrics[f"prefix_{k}_ori_loss"] = masked_mean(
                    action_loss[:, :k, 3:7],
                    valid[:, :k, 3:7],
                )
                metrics[f"prefix_{k}_gripper_loss"] = masked_mean(
                    action_loss[:, :k, -1:],
                    valid[:, :k, -1:],
                )

        # ------------------------------------------------------------
        # Per-timestep loss:
        # Diagnose whether sample error accumulates over time.
        # If t00 is low but t15/t31 are high, then long-horizon
        # accumulation is severe.
        # ------------------------------------------------------------
        num_steps = action_loss.shape[1]

        for t in range(num_steps):
            metrics[f"t{t:02d}_action_loss"] = masked_mean(
                action_loss[:, t:t + 1, :],
                valid[:, t:t + 1, :],
            )
            metrics[f"t{t:02d}_pos_loss"] = masked_mean(
                action_loss[:, t:t + 1, :3],
                valid[:, t:t + 1, :3],
            )
            metrics[f"t{t:02d}_ori_loss"] = masked_mean(
                action_loss[:, t:t + 1, 3:7],
                valid[:, t:t + 1, 3:7],
            )
            metrics[f"t{t:02d}_gripper_loss"] = masked_mean(
                action_loss[:, t:t + 1, -1:],
                valid[:, t:t + 1, -1:],
            )

        return metrics

    def _compute_loss(self, batch_input: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a_hat, a_gt, is_pad, latent_losses = self.forward(batch_input, training=True)
        if a_gt is None or is_pad is None or latent_losses is None:
            raise ValueError("Ground truth actions, is_pad, latent_losses are required in training mode.")
        if self.loss_type == "l1":
            action_loss = F.l1_loss(a_hat, a_gt, reduction="none")
        elif self.loss_type == "mse":
            action_loss = F.mse_loss(a_hat, a_gt, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # is_pad: [B, T]
        # True  = padding, should be ignored
        # False = valid action
        valid = (~is_pad.bool()).unsqueeze(-1).expand_as(action_loss)

        action_loss = action_loss.masked_fill(~valid, 0.0)

        def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return loss.sum() / mask.sum().clamp_min(1)

        loss_dict = {
            "action_loss": masked_mean(action_loss, valid),
            "traj_loss": masked_mean(action_loss[:, 1:, :], valid[:, 1:, :]),
            "traj_pos_loss": masked_mean(action_loss[:, 1:, :3], valid[:, 1:, :3]),
            "traj_ori_loss": masked_mean(action_loss[:, 1:, 3:7], valid[:, 1:, 3:7]),
            "traj_gripper_loss": masked_mean(action_loss[:, 1:, -1], valid[:, 1:, -1]),
        }

        #TODO 这里逻辑不知道是否正确
        per_level = []
        for i, l in enumerate(latent_losses):
            if l is None:
                continue
            l_scalar = l.mean()  # l 可能是 [B, seq_len, H] 或 [B, seq_len]
            loss_dict[f"latent_loss_l{i}"] = l_scalar
            per_level.append(l_scalar)
        if per_level:
            latent_total = sum(per_level) / len(per_level)
            loss_dict["latent_loss"] = latent_total

        total_loss = loss_dict["action_loss"] + 1 * loss_dict["latent_loss"]
        return total_loss, loss_dict

    def update(self, batch_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.training_mode(training=True)
        total_loss, loss_dict = self._compute_loss(batch_input)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=-100.0)

        self.opt.zero_grad(set_to_none=True)
        self.accelerator.backward(total_loss)
        if self.actor_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.actor_grad_clip)
        self.opt.step()
        if hasattr(self, "lr_scheduler"):
            self.lr_scheduler.step()

        return {
            "total_loss": total_loss.detach(),
            **{k: v.detach() for k, v in loss_dict.items()}
        }
