# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention:
        See TransformerDecoderLayer + TransformerEncoderLayer
    * extra LN at the end of encoder is removed: See Transformer
    * decoder returns a stack of activations from all decoding layers:
        See TransformerDecoder
"""

from typing import Optional

import torch
from torch.nn.modules.transformer import _get_clones
from torch import nn, Tensor
from src.methods.utils import ImgChLayerNorm, layernorm_for_cnn, identity_cls

class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        nmtpheads=None,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first=False,
        return_intermediate_dec=False,

    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first
        )
        # NOTE: Original implementation always have nn.LayerNorm here
        encoder_norm = nn.LayerNorm(d_model) if norm_first else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, norm_first
        )
        decoder_norm = nn.LayerNorm(d_model)
        if nmtpheads > 0:
            mtp_head = MTPHeadLayer(
                d_model, nhead, dim_feedforward, dropout, activation, norm_first
            )
        else:
            mtp_head = None
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            mtp_head=mtp_head,
            num_mtp=nmtpheads,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        obs_feat,
        tgt_pos_embed=None,
        mem_pos_embed=None,
        latent_embed=None,
        proprio_embed=None,
        proprio=None,
        task_embed=None,
        actions=None,
        action_head=None,
        de_action_head=None,
        training=True,
        action_seq=1,
        is_pad=None,
        action_order="FORWARD",
        execute_threshold=0.01,
        execution_length=100,
    ):
        # TODO flatten only when input has H and W
        assert len(obs_feat.shape) == 3
        # flatten NxHWxC to HWxNxC
        l ,bs, c = obs_feat.shape
        obs_feat = torch.cat([obs_feat, proprio_embed], axis=0) 
        
        # query_embed = query_embed.permute(1,0,2).repeat(1, bs, 1)
        tgt_pos_embed = tgt_pos_embed.unsqueeze(1).repeat(1, bs, 1)
            

        # tgt = torch.zeros_like(query_embed)
        memory = self.encoder(obs_feat, pos=mem_pos_embed)
        output = self.decoder(
            obs_feat,
            actions,
            memory=memory,
            mem_pos_embed=mem_pos_embed,
            tgt_pos_embed=tgt_pos_embed,
            training=training,
            action_seq=action_seq,
            action_head=action_head,
            de_action_head=de_action_head,
            proprio_embed=proprio_embed,
            proprio=proprio,
            is_pad=is_pad,
            action_order=action_order,
            execute_threshold=execute_threshold,
            execution_length=execution_length,
        )

        return output.transpose(0, 1)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    # NOTE: Only difference is passing the pos parameter
    # to the forward to be passed to each layer
    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, mtp_head=None, num_mtp=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        if mtp_head is not None:
            self.mtp_layer = _get_clones(mtp_head, num_mtp)
        else:
            self.mtp_layer = None
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        # self.de_action_head = nn.Linear(8, hidden_dim)
        self.sos_embedding = nn.Parameter(torch.randn(1,1, 512))

    def forward(
        self,
        tgt,
        actions,
        memory,
        mem_pos_embed: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
        training: Optional[bool] = False,
        action_seq: Optional[int] = 1,
        action_head: Optional[nn.Module] = None,
        de_action_head: Optional[nn.Module] = None,
        proprio_embed: Optional[Tensor] = None,
        proprio: Optional[Tensor] = None,
        is_pad: Optional[Tensor] = None,
        action_order: Optional[str] = "FORWARD",
        execute_threshold: Optional[float] = 0.01,
        execution_length: Optional[int] = 100,

    ):

        if training:  # training=True时使用teacher forcing

            # 简化版本：只支持abs模式
            # 如果actions是4维的，说明是MTP（多时间步预测）格式，只取第一个时间步
            if len(actions.shape) == 4:
                # 形状是 [batch, seq, mtp_size, action_dim]，只取第一个mtp头
                actions = actions[:, :, 0, :]  # [batch, seq, action_dim]
            # 如果is_pad是3维的，同样只取第一个mtp头
            if is_pad is not None and len(is_pad.shape) == 3:
                # 形状是 [batch, seq, mtp_size]，只取第一个mtp头
                is_pad = is_pad[:, :, 0]  # [batch, seq]
            current_input = de_action_head(actions)
            current_input = current_input.permute(1, 0, 2)
            current_input = torch.concat([self.sos_embedding.repeat(1,current_input.shape[1],1), current_input[:-1]])
            
            length = actions.shape[1]
            tgt_mask = ~(torch.triu(torch.ones(length, length, device=current_input.device)) == 1).transpose(0, 1)
            # 逐层处理
            # print(torch.norm(current_input[0]))
            for layer in self.layers:
                current_input = layer(
                    current_input,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=is_pad,
                    # memory_key_padding_mask=memory_key_padding_mask,
                    mem_pos_embed=mem_pos_embed,
                    tgt_pos_embed=tgt_pos_embed,
                    )
            
            # print("train1",torch.norm(current_input[0]))
            if self.mtp_layer is not None:
                mtp_token_list = []
                for layer in self.mtp_layer:
                    mtp_token_list.append(layer(
                    current_input,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    tgt_key_padding_mask=is_pad,
                    # memory_key_padding_mask=memory_key_padding_mask,
                    mem_pos_embed=mem_pos_embed,
                    tgt_pos_embed=tgt_pos_embed,
                    ))
                current_input = torch.stack(mtp_token_list, dim=-2)


            # print("train2",torch.norm(current_input[0,0,0]))
            # 正则化
            if self.norm is not None:
                output = self.norm(current_input)
            else:
                output = current_input
            
            # print(f"time cost: {time.time() - t0}")
            return output

        else: # training=False时进行auto-regressive inference

            current_input = self.sos_embedding.repeat(1,memory.shape[1],1)
            # current_input = torch.cat([current_input, de_action_head(tgt)], dim=0)

            # 逐步生成 action_seq 个后续步骤
            for step in range(action_seq):
                length = current_input.shape[0]
                tgt_mask = ~(torch.triu(torch.ones(length, length, device=current_input.device)) == 1).transpose(0, 1)
                tgt_pos_embed_tmp = tgt_pos_embed[:current_input.shape[0],...]
                # 通过 Transformer 的多层进行编码
                x = current_input
                # if step == 0:
                #     print(torch.norm(x[0]))
                for layer in self.layers:
                    x = layer(
                        x, 
                        memory, 
                        tgt_mask=tgt_mask, 
                        memory_mask=None, 
                        mem_pos_embed=mem_pos_embed, 
                        tgt_pos_embed=tgt_pos_embed_tmp,
                    )

                if self.mtp_layer is not None:
                    x = self.mtp_layer[0](
                    x,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    mem_pos_embed=mem_pos_embed,
                    tgt_pos_embed=tgt_pos_embed_tmp,
                    )
                

                if self.norm is not None:
                    x = self.norm(x)

                next_action_embed = x[-1]  # 取最后一个时间步的输出 [B, D]

                current_input = torch.cat([current_input, next_action_embed.unsqueeze(0)], dim=0)
                
                # 简化的停止条件判断
                if action_order == 'REVERSE':
                    v_sp = (action_head(current_input[1]) - proprio)[...,:3]
                    v_sc = (action_head(current_input[1]) - action_head(next_action_embed))[...,:3]
                    dist_sp = torch.norm(v_sp, dim=-1) # start to postion state
                    dist_sc = torch.norm(v_sc, dim=-1) # start to current prediction
                    # print(dist_sp - dist_sc, dist_sp1 - dist_sc1)
                    if ((dist_sp - dist_sc) < execute_threshold).all() and ((v_sp * v_sc).sum(dim=-1) >= 0).all():
                        break
                elif action_order == 'HYBRID':
                    v_proprio_nbp = (action_head(current_input[1]) - proprio)[...,:3]
                    v_proprio_current = (action_head(next_action_embed) - proprio)[...,:3]
                    dist_proprio_nbp = torch.norm(v_proprio_nbp, dim=-1) # start to postion state
                    dist_proprio_current = torch.norm(v_proprio_current, dim=-1) # start to current prediction
                    # print(dist_sp - dist_sc, dist_sp1 - dist_sc1)
                    if step>0 and ((dist_proprio_nbp - dist_proprio_current) < execute_threshold).all() and ((v_proprio_nbp * v_proprio_current).sum(dim=-1) >= 0).all():
                        break
                elif action_order == 'FORWARD':
                    if step + 1 == execution_length:
                        break
            current_input = current_input.unsqueeze(-2)
            return current_input[1:,...]

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn_from_str(activation)()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                pos,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, pos))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos,
    ) -> Tensor:
        q = k = self.with_pos_embed(x, pos)
        # NOTE: Order is different in original implementation x, x, x
        x = self.self_attn(
            q,
            k,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn_from_str(activation)()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        mem_pos_embed: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block( # self-attention
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_pos_embed,
            )
            x = x + self._mha_block( # cross-attention
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                mem_pos_embed,
                tgt_pos_embed,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_pos_embed)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, mem_pos_embed, tgt_pos_embed
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query_pos,
    ) -> Tensor:
        q = k = self.with_pos_embed(x, query_pos)
        # NOTE: Order is different in original implementation x, x, x
        x = self.self_attn(
            q, k, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos,
        query_pos,
    ) -> Tensor:
        # NOTE: Order is different in original implementation x, mem, mem
        x = self.multihead_attn(
            self.with_pos_embed(x, query_pos),
            self.with_pos_embed(mem, pos),
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return self.dropout2(x)

    # feedforward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MTPHeadLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first: bool = False,

    ):
        super().__init__(d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        norm_first= False)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.linear3 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn2 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_fn_from_str(activation)()
        self.norm_first = norm_first

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        mem_pos_embed: Optional[Tensor] = None,
        tgt_pos_embed: Optional[Tensor] = None,
    ):
        x = tgt
        if self.norm_first:
            x = x + self._sa_block( # self-attention
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_pos_embed,
            )
            x = x + self._mha_block( # cross-attention
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                mem_pos_embed,
                tgt_pos_embed,
            )
            x = x + self._ff_block2(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_pos_embed)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, mem_pos_embed, tgt_pos_embed
                )
            )
            x = self.norm3(x + self._ff_block2(x))

        return x


        # feedforward block
    def _ff_block2(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout_ffn1(self.activation(self.linear1(x))))
        x = self.linear4(self.dropout_ffn2(self.activation(self.linear3(x))))
        return self.dropout3(x)


def get_activation_fn_from_str(act: str) -> type[nn.Module]:
    if act == "relu":
        return nn.ReLU
    elif act == "lrelu":
        return nn.LeakyReLU
    elif act == "elu":
        return nn.ELU
    elif act == "tanh":
        return nn.Tanh
    elif act == "prelu":
        return nn.PReLU
    elif act == "silu":
        return nn.SiLU
    elif act == "gelu":
        return nn.GELU
    elif act == "glu":
        return nn.GLU
    else:
        raise ValueError("%s not recognized." % act)


def get_normalization_fn_from_str(norm: str) -> type[nn.Module]:
    if norm == "layer":
        return nn.LayerNorm
    elif norm == "layer_for_cnn":
        return layernorm_for_cnn
    elif norm == "img_ch_layer":
        return ImgChLayerNorm
    elif norm == "group":
        return nn.GroupNorm
    elif norm == "batch1d":
        return nn.BatchNorm1d
    elif norm == "batch2d":
        return nn.BatchNorm2d
    elif norm == "identity":
        return identity_cls
    else:
        raise ValueError("%s not recognized." % norm)
