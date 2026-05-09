# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from re import X
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.envs.rlbench.wrappers.rescale_from_tanh import MinMaxNorm
from src.methods.base import BaseMethod, BatchedActionSequence
from src.methods.backbone import build_backbone
from src.methods.coa.transformer import (
    Transformer,
)
from typing import Tuple, List, Optional, Union, Any
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.optimizer import AcceleratedOptimizer
from transformers.optimization import get_scheduler
from hydra.utils import instantiate
from src.methods.utils import extract_many_from_batch, flatten_time_dim_into_channel_dim, stack_tensor_dictionary, extract_from_spec
import numpy as np
import torchvision.transforms as tvf




class ImageEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim, position_embedding, lr_backbone, masks, backbone, dilation, use_lang_cond, use_frozen_bn=False):
        super().__init__()
        assert (len(input_shape) == 4), f"Expected shape (View, C, H, W), but got {input_shape}"
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
        
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, task_emb: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the image encoder.

        Args:
            x: Input tensor. shape: (b, v, c, h, w)
            task_emb: Task embedding. shape: (b, task_emb_dim)

        Returns:
            img_feat: Image features. shape: (b, c, v, l)
            pos: Positional encodings. shape: (b, c, v, l)
        '''
        assert (
            self._input_shape == x.shape[1:]
        ), f"expected input shape {self._input_shape} but got {x.shape[1:]}"

        all_cam_features = []
        all_cam_pos = []
        shape = x.shape
        for cam_id in range(self._input_shape[0]):
            # (b, v, fs, c, h, w) -> (b*fs, c, h, w)
            cur_x = x[:, cam_id].reshape(-1, 3, *self._input_shape[2:])

            # feat: (b*fs, c, h, w) -> (b*fs, feat_dim, 3, 3)
            feat, pos = self.backbone(cur_x)

            # feat: (b*fs, feat_dim, 3, 3) -> (b*fs, hidden_dim, 3, 3)
            feat = self.input_proj(feat[0])
            # pos: (b, pos_feat_dim, 3, 3)
            pos = pos[0]

            all_cam_features.append(feat)
            all_cam_pos.append(pos)

        # (b*fs, hidden_dim, 3, 3) -> (b*fs, hidden_dim, 3, 3*v)
        img_feat = torch.cat(all_cam_features, dim=3)

        # (b*fs, hidden_dim, 3, 3) -> (b, fs*hidden_dim, 3, 3*v)
        img_feat = img_feat.reshape(shape[0], -1, *img_feat.shape[2:])

        # (b, pos_feat_dim, 3, 3*v)
        pos = torch.cat(all_cam_pos, dim=3)

        return img_feat, pos
        
class ActorModel(nn.Module):
    def __init__(self,         
        hidden_dim: int = 512,
        dropout: float = 0.1,
        nheads: int = 8,
        nmtpheads: int=0,
        dim_feedforward: int = 3200,
        enc_layers: int = 4,
        dec_layers: int = 6,
        pre_norm: bool = True,
        state_dim: int = 8,
        action_dim: int = 8,
        num_queries: int = 100,
        use_lang_cond: bool = False,
        action_order: str = "FORWARD",
        execute_threshold: float = 0.01,
        execution_length: int = 100,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Actor model for CoA policy. It takes image features and other raw inputs to generate action sequence.

        Args:
            hidden_dim: Hidden dimension of the model
            dropout: Dropout rate
            nheads: Number of attention heads
            nmtpheads: Number of multi-head attention heads
            dim_feedforward: Dimension of the feedforward network
            enc_layers: Number of encoder layers
            dec_layers: Number of decoder layers
            pre_norm: Whether to use pre-norm
            state_dim: Dimension of the state (proprioception)
            action_dim: Dimension of the action
            num_queries: Number of action queries
            use_lang_cond: Whether to use language conditioning
            action_order: Action order (FORWARD/REVERSE/HYBRID)
            execute_threshold: Execution threshold for stopping
            execution_length: Execution length
        '''
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Save configuration parameters
        self.action_order = action_order
        self.execute_threshold = execute_threshold
        self.execution_length = execution_length

        # Transformer backbone
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            nmtpheads=nmtpheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=pre_norm,
            return_intermediate_dec=True,
        )

        # Action embedding layers 
        self.action2embed = nn.Linear(action_dim, hidden_dim)  
        self.embed2action = nn.Linear(hidden_dim, action_dim)   

        # Position embeddings for action queries
        self.register_buffer('queries_pos_embed', 
            self._build_sinusoidal_pos_embed(num_queries, hidden_dim))
        
        # Position embeddings for action queries for memory: proprio
        self.learnable_mem_pos_embed = nn.Parameter(
            torch.randn(1, 1, hidden_dim))  # Only provide position encoding for proprio embedding

    def _build_sinusoidal_pos_embed(self, num_pos: int, hidden_dim: int) -> torch.Tensor:
        # build sinusoidal position embeddings for action queries
        position = torch.arange(num_pos).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           -(np.log(10000.0) / hidden_dim))
        
        pos_embed = torch.zeros(1, num_pos, hidden_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)

        pos_embed.requires_grad_(False)
        return pos_embed

    def forward(
        self,
        obs_feat: Tuple[torch.Tensor, torch.Tensor],
        proprio: torch.Tensor,
        task_embed: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the actor model.

        Args:
            obs_feat (Tuple[torch.Tensor, torch.Tensor]):
                    Image features and positional encodings. shape: ((b, c, v, l), (b, c, v, l))
            proprio (torch.Tensor): Tensor containing proprioception features. shape: (b, d)
            task_embed (torch.Tensor, optional): Tensor containing task embedding.
            actions (torch.Tensor, optional): Tensor containing ground truth action sequences.
            is_pad (torch.Tensor, optional): Tensor indicating ground truth padding positions.
            training (bool): Whether in training mode. True for training, False for inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: 
                - a_hat: Predicted actions (b, num_queries, action_dim)
                - x_hat: Predicted latent embeddings
                - x_gt: Ground truth action embeddings (for training, None for inference)
        """
        
        # Extract image features and position embeddings
        img_feat, img_pos_embed = obs_feat
        bs = img_feat.shape[0]
        
        # Reshape image features to sequence format: (b, c, v, l) -> (l*v, b, c)
        seq_len = img_feat.shape[2] * img_feat.shape[3]  # v * l
        img_feat = img_feat.flatten(2).permute(2, 0, 1)
        img_pos_embed = img_pos_embed.flatten(2).permute(2, 0, 1)
        
        # Add memory position embeddings for proprio to img_pos_embed
        mem_pos_embed = self.learnable_mem_pos_embed
        pos_embed = torch.cat([img_pos_embed, mem_pos_embed], dim=0)

        # Proprioception embedding
        proprio_embed = self.action2embed(proprio)
        # b,l,d -> l,b,d
        proprio_embed = proprio_embed.permute(1, 0, 2) 

        
        # Call transformer 
        hs = self.transformer(
            obs_feat=img_feat,
            actions=actions,
            mem_pos_embed=pos_embed,
            tgt_pos_embed=self.queries_pos_embed.squeeze(0),  # (num_queries, hidden_dim)
            latent_embed=None,
            proprio_embed=proprio_embed,
            proprio=proprio,
            task_embed=task_embed,
            action_head=self.embed2action,
            de_action_head=self.action2embed,
            training=training,
            action_seq=self.num_queries,
            is_pad=is_pad,
            action_order=self.action_order,
            execute_threshold=self.execute_threshold,
            execution_length=self.execution_length,
        )
        
        # Generate predictions
        a_hat = self.embed2action(hs)
        x_hat = hs
        
        # Handle ground truth actions for training
        if actions is not None:
            # Keep actions in original dimensions, let x_gt match x_hat dimensions
            x_gt = self.action2embed(actions)
        else:
            x_gt = None
            
        return a_hat, x_hat, x_gt


class CoA(BaseMethod):
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
        execute_threshold=0.01,
        execution_length=100,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        '''
        CoA policy. It takes image features and other raw inputs to generate action sequence.

        Args:
            encoder_model: Encoder model
            actor_model: Actor model
            action_order: Organization order of the action sequence, related to dynamic stop
            action_mode: Mode of the action, related to dynamic stop
            loss_type: Type of the loss
            latent_loss_type: Type of the latent loss
            execute_threshold: Execution threshold for stopping
            execution_length: Execution length
            *args, **kwargs: Additional arguments   
        '''
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.adaptive_lr = adaptive_lr
        self.weight_decay = weight_decay
        self.num_train_steps = num_train_steps
        self.use_lang_cond = use_lang_cond
        self.action_order = action_order
        self.action_mode = action_mode
        self.loss_type = loss_type
        self.latent_loss_type = latent_loss_type
        self.execute_threshold = execute_threshold
        self.execution_length = execution_length
        self.actor_grad_clip = actor_grad_clip

        # Get device information
        self.device = self.accelerator.device if self.accelerator else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_model = encoder_model()
        self.actor_model = actor_model(
            action_order=action_order,
            execute_threshold=execute_threshold, 
            execution_length=execution_length
        )
        
        # Move models to device
        self.encoder_model = self.encoder_model.to(self.device)
        self.actor_model = self.actor_model.to(self.device)
        
        VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
        VISUAL_OBS_STD = [0.229, 0.224, 0.225]
        self.img_normalizer = tvf.Normalize(
            mean=VISUAL_OBS_MEAN, std=VISUAL_OBS_STD
        )

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

        self.opt = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.adaptive_lr:
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.opt,
                num_warmup_steps=100,
                num_training_steps=self.num_train_steps,
            )


        self.prepare_accelerator()

    def forward(
        self,
        batch_input: dict[str, torch.Tensor],
        training: bool = True
    ) -> Union[BatchedActionSequence, Tuple[Any, ...]]:
        '''
        Forward pass of the CoA policy.

        Args:
            batch_input: Input batch data
            training: Whether in training mode

        Returns:
            Tuple containing action predictions and other intermediate results
        '''
        raw_img = extract_many_from_batch(batch_input, 'rgb')
        img = flatten_time_dim_into_channel_dim(stack_tensor_dictionary(raw_img, dim=1))
        proprio = batch_input['low_dim_state']
        if training:
            a_gt = batch_input['action']
            is_pad = batch_input['is_pad']
        else:
            a_gt = None
            is_pad = None
            
        if 'task_emb' in batch_input:
            task_emb = batch_input['task_emb']
        else:
            task_emb = None

        # normalize img
        img = self.img_normalizer(img/255)
        # encoder forward
        obs_feat = self.encoder_model(img) # obs_feat: img_feat, pos_embed
        # actor forward
        a_hat, x_hat, x_gt = self.actor_model(obs_feat, proprio, task_emb, a_gt, is_pad, training=training)
        return a_hat, a_gt, x_hat, x_gt, is_pad, proprio, task_emb

    def training_mode(self, training: bool = True):
        if training:
            self.actor_model.train()
        else:
            self.actor_model.eval()

    @torch.no_grad()
    def act(self, batch_input: dict[str, torch.Tensor]) -> BatchedActionSequence:
        '''
        Generate action sequence for inference with action order handling.
        Args:
            batch_input: Input observations
        Returns:
            action_sequence: Generated action sequence with proper ordering, shape: (batch, seq, action_dim)
        '''
        self.training_mode(training=False)
        result = self.forward(batch_input, training=False)
        a_hat = result[0]
        # [batch, seq, mtp_size, action_dim]
        if len(a_hat.shape) == 4:
            a_hat = a_hat[:, :, 0, :]
        if self.action_order == "REVERSE":
            a_hat = torch.flip(a_hat, dims=[1])
        elif self.action_order == "HYBRID":
            a_hat = torch.roll(a_hat, shifts=-1, dims=1)

        # dummy_action = torch.tensor([0,0,1,0,0,0,1,1], device=a_hat.device).unsqueeze(0).unsqueeze(0)
        return a_hat
        # return dummy_action
    
    def validate(
            self,
            batch_input: dict[str, torch.Tensor],
            use_generated_actions: bool = True,
        ) -> dict[str, torch.Tensor]:
            """
            Offline validation for CoA.

            use_generated_actions=True:
                obs -> CoA inference path -> predicted action sequence
                compare predicted actions with GT actions.

            use_generated_actions=False:
                teacher-forcing path:
                obs + GT actions -> predicted actions
                compare predicted actions with GT actions.
            """
            self.training_mode(training=False)

            a_gt = batch_input["action"]
            is_pad = batch_input["is_pad"]

            if use_generated_actions:
                # Important:
                # Do NOT call self.act() here, because self.act() flips/rolls actions
                # for environment execution when action_order is REVERSE/HYBRID.
                # The dataset GT for CoA is already stored in the internal training order,
                # so we compare raw inference output against raw GT.
                a_hat, _, _, _, _, _, _ = self.forward(
                    batch_input,
                    training=False,
                )
                x_hat, x_gt = None, None
            else:
                # Teacher-forcing validation.
                a_hat, a_gt, x_hat, x_gt, is_pad, _, _ = self.forward(
                    batch_input,
                    training=True,
                )

            # CoA usually uses MTP format:
            #   a_hat: [B, T, M, D]
            #   a_gt:  [B, T, M, D]
            #   is_pad:[B, T, M]
            #
            # But make this robust in case inference returns [B, T, D].
            if a_hat.ndim == 3 and a_gt.ndim == 4:
                a_gt_cmp = a_gt[:, :, 0, :]
                is_pad_cmp = is_pad[:, :, 0]
            else:
                a_gt_cmp = a_gt
                is_pad_cmp = is_pad

            if self.loss_type == "l1":
                raw_action_loss = F.l1_loss(a_hat, a_gt_cmp, reduction="none")
            elif self.loss_type == "mse":
                raw_action_loss = F.mse_loss(a_hat, a_gt_cmp, reduction="none")
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            valid = (~is_pad_cmp.bool()).unsqueeze(-1).expand_as(raw_action_loss)
            action_loss = raw_action_loss.masked_fill(~valid, 0.0)

            def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return loss.sum() / mask.sum().clamp_min(1)

            metrics = {
                "action_loss": masked_mean(action_loss, valid),
            }

            # MTP case: [B, T, M, D]
            if action_loss.ndim == 4:
                main_loss = action_loss[:, :, 0, :]
                main_valid = valid[:, :, 0, :]

                metrics.update({
                    # first query / key-frame action
                    "kfa_loss": masked_mean(main_loss[:, :1, :], main_valid[:, :1, :]),
                    "kfa_pos_loss": masked_mean(main_loss[:, :1, :3], main_valid[:, :1, :3]),
                    "kfa_ori_loss": masked_mean(main_loss[:, :1, 3:7], main_valid[:, :1, 3:7]),
                    "kfa_gripper_loss": masked_mean(main_loss[:, :1, -1:], main_valid[:, :1, -1:]),

                    # remaining trajectory actions, first MTP head only
                    "traj_loss": masked_mean(main_loss[:, 1:, :], main_valid[:, 1:, :]),
                    "traj_pos_loss": masked_mean(main_loss[:, 1:, :3], main_valid[:, 1:, :3]),
                    "traj_ori_loss": masked_mean(main_loss[:, 1:, 3:7], main_valid[:, 1:, 3:7]),
                    "traj_gripper_loss": masked_mean(main_loss[:, 1:, -1:], main_valid[:, 1:, -1:]),
                })
            else:
                # Non-MTP fallback: [B, T, D]
                metrics.update({
                    "kfa_loss": masked_mean(action_loss[:, :1, :], valid[:, :1, :]),
                    "kfa_pos_loss": masked_mean(action_loss[:, :1, :3], valid[:, :1, :3]),
                    "kfa_ori_loss": masked_mean(action_loss[:, :1, 3:7], valid[:, :1, 3:7]),
                    "kfa_gripper_loss": masked_mean(action_loss[:, :1, -1:], valid[:, :1, -1:]),

                    "traj_loss": masked_mean(action_loss[:, 1:, :], valid[:, 1:, :]),
                    "traj_pos_loss": masked_mean(action_loss[:, 1:, :3], valid[:, 1:, :3]),
                    "traj_ori_loss": masked_mean(action_loss[:, 1:, 3:7], valid[:, 1:, 3:7]),
                    "traj_gripper_loss": masked_mean(action_loss[:, 1:, -1:], valid[:, 1:, -1:]),
                })

            # Optional latent validation.
            # Only available in teacher-forcing mode because x_gt requires GT actions.
            if (not use_generated_actions) and (x_hat is not None) and (x_gt is not None):
                if self.latent_loss_type == "l1":
                    raw_latent_loss = F.l1_loss(x_hat, x_gt, reduction="none")
                elif self.latent_loss_type == "mse":
                    raw_latent_loss = F.mse_loss(x_hat, x_gt, reduction="none")
                else:
                    raise ValueError(f"Unknown latent_loss_type: {self.latent_loss_type}")

                # x_hat/x_gt hidden dim usually corresponds to action positions.
                latent_valid = (~is_pad.bool()).unsqueeze(-1).expand_as(raw_latent_loss)
                latent_loss = raw_latent_loss.masked_fill(~latent_valid, 0.0)
                metrics["latent_loss"] = masked_mean(latent_loss, latent_valid)

            return metrics
    

    def _compute_loss(self, batch_input: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Compute the loss of the CoA policy.
        Args:
            batch_input: Observations
        Returns:
            action_loss, latent_loss, total_loss: 各项损失
        '''
        result = self.forward(batch_input, training=True)
        a_hat, a_gt, x_hat, x_gt, is_pad, proprio, task_embed = result
        # action loss compute
        if self.loss_type == 'l1':
            action_loss = F.l1_loss(a_hat, a_gt, reduction='none')
        elif self.loss_type == 'mse':
            action_loss = F.mse_loss(a_hat, a_gt, reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        action_loss = action_loss * ~is_pad.unsqueeze(-1)
        # print(a_hat[0,0,0,2], a_gt[0,0,0,2])
        # latent loss compute
        if x_gt is not None:
            if self.latent_loss_type == 'l1':
                latent_loss = F.l1_loss(x_hat, x_gt, reduction='none')
            elif self.latent_loss_type == 'mse':
                latent_loss = F.mse_loss(x_hat, x_gt, reduction='none')
            else:
                raise ValueError(f"Unknown latent_loss_type: {self.latent_loss_type}")
        else:
            latent_loss = torch.tensor(0.0, device=a_hat.device)
        latent_loss = latent_loss * ~is_pad.unsqueeze(-1)
        # Compute various loss components
        loss_dict = {
            # Compute main losses
            "action_loss": action_loss.sum() / (action_loss != 0).sum(),
            "latent_loss": latent_loss.sum() / (latent_loss != 0).sum(),
            
            # KFA (Key frame action) losses - first action
            "kfa_loss": action_loss[:,0,0,:].mean(),
            "kfa_pos_loss": action_loss[:,0,0,:3].mean(),
            "kfa_ori_loss": action_loss[:,0,0,3:7].mean(),
            "kfa_gripper_loss": action_loss[:,0,0,-1].mean(),
            
            # Trajectory losses - remaining actions
            "traj_loss": action_loss[:,1:,0,:].sum() / (action_loss[:,1:,0,:]!=0).sum(),
            "traj_pos_loss": action_loss[:,1:,0,:3].sum() / (action_loss[:,1:,0,:3]!=0).sum(),
            "traj_ori_loss": action_loss[:,1:,0,3:7].sum() / (action_loss[:,1:,0,3:7]!=0).sum(),
            "traj_gripper_loss": action_loss[:,1:,0,-1].sum() / (action_loss[:,1:,0,-1]!=0).sum(),
        }


        total_loss = loss_dict["action_loss"] + loss_dict["latent_loss"]
        
        return total_loss, loss_dict

    def update(self, batch_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        '''
        Update the CoA policy. it calls forward function and compute the loss.
        Args:
            batch_input: Observations
        Returns:
            dict: {"total_loss": ..., "action_loss": ..., "latent_loss": ...}
        '''
        self.training_mode(training=True)
        total_loss, loss_dict = self._compute_loss(batch_input)
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=-100.0)
        self.opt.zero_grad(set_to_none=True)
        self.accelerator.backward(total_loss)
        if self.actor_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.actor_grad_clip)
        self.opt.step()
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        return {
            "total_loss": total_loss.detach(),
            **{k: v.detach() for k, v in loss_dict.items()}
        }




@hydra.main(
    config_path="../../src/cfgs", config_name="launch.yaml", version_base=None
)

def main(cfg: DictConfig):
    print("Starting main function...")
    import pickle
    print("Loading batch data...")
    with open('./batch.pkl', 'rb') as f:
        loaded_batch = pickle.load(f)
    loaded_batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in loaded_batch.items()}
    print("Batch data loaded successfully")

    print("Initializing CoA model...")
    coa = hydra.utils.instantiate(cfg.method).cuda()
    print("CoA model initialized")

    print("Starting training loop...")
    num_iterations = 100  # Set number of iterations
    
    for iteration in range(num_iterations):
        result = coa.update(loaded_batch)
        total_loss = result["total_loss"].item()
        
        # Print loss every 10 iterations
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {total_loss:.6f}")
        
        # Record loss every time but don't print (avoid too much output)
        if iteration == 0:
            print(f"Initial Loss: {total_loss:.6f}")
    
    print(f"Training completed. Final Loss: {total_loss:.6f}")


if __name__ == "__main__":
    main()
