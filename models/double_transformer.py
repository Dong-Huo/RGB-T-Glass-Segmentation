# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.position_encoding import PositionEmbeddingLearned
from util.misc import NestedTensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # encoder
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        linear_layer = nn.Linear(d_model, 1)

        self.fusion_encoder = FusionEncoder(encoder_layer, linear_layer, num_encoder_layers, encoder_norm)

        self.embedding_generator = PositionEmbeddingLearned(d_model // 2)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, rgb_src, rgb_mask, rgb_pos_embed, temperature_src, temperature_mask, temperature_pos_embed,
                gate_flag):
        # flatten NxCxHxW to HWxNxC

        bs, c, h, w = rgb_src.shape

        fusion_pos_embed = self.embedding_generator(
            NestedTensor(torch.cat([rgb_src, temperature_src], 2), torch.cat([rgb_mask, temperature_mask], 1)))

        # rgb_branch
        rgb_src = rgb_src.flatten(2).permute(2, 0, 1)
        rgb_pos_embed = rgb_pos_embed.flatten(2).permute(2, 0, 1)
        fusion_pos_embed = fusion_pos_embed.flatten(2).permute(2, 0, 1)
        rgb_mask = rgb_mask.flatten(1)

        # temperature_branch
        temperature_src = temperature_src.flatten(2).permute(2, 0, 1)
        temperature_pos_embed = temperature_pos_embed.flatten(2).permute(2, 0, 1)
        temperature_mask = temperature_mask.flatten(1)

        fusion_memory = self.fusion_encoder(rgb_src, temperature_src, memory_key_padding_mask=rgb_mask,
                                            rgb_pos=rgb_pos_embed, temperature_pos=temperature_pos_embed,
                                            fusion_pos=fusion_pos_embed, gate_flag=gate_flag)

        fusion_memory = fusion_memory.permute(1, 2, 0).view(bs, -1, h, w)

        return fusion_memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class FusionEncoder(nn.Module):
    def __init__(self, encoder_layer, Linear_layers, num_layers, norm=None):
        super().__init__()
        self.rgb_layers = _get_clones(encoder_layer, num_layers)
        self.temperature_layers = _get_clones(encoder_layer, num_layers)
        self.fusion_layers = _get_clones(encoder_layer, num_layers)
        self.attn_layers = _get_clones(encoder_layer, num_layers)
        self.Linear_layers = _get_clones(Linear_layers, num_layers)

        self.num_layers = num_layers
        self.norm = norm

    def forward(self, rgb_memory, temperature_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                rgb_pos: Optional[Tensor] = None,
                temperature_pos: Optional[Tensor] = None,
                fusion_pos: Optional[Tensor] = None,
                gate_flag: bool = True):

        s, b, c = rgb_memory.shape

        fusion_key_padding_mask = torch.cat([memory_key_padding_mask, memory_key_padding_mask], 1)

        for i in range(self.num_layers):
            rgb_memory = self.rgb_layers[i](rgb_memory, src_mask=memory_mask,
                                            src_key_padding_mask=memory_key_padding_mask, pos=rgb_pos)

            temperature_memory = self.temperature_layers[i](temperature_memory, src_mask=memory_mask,
                                                            src_key_padding_mask=memory_key_padding_mask,
                                                            pos=temperature_pos)

            fusion_memory = torch.cat([rgb_memory, temperature_memory], 0)

            fusion_memory = self.fusion_layers[i](fusion_memory, src_mask=memory_mask,
                                                  src_key_padding_mask=fusion_key_padding_mask,
                                                  pos=fusion_pos)

            attention_memory = self.attn_layers[i](fusion_memory, src_mask=memory_mask,
                                                   src_key_padding_mask=fusion_key_padding_mask,
                                                   pos=fusion_pos)

            attention_weights = torch.sigmoid(self.Linear_layers[i](attention_memory))

            if gate_flag:
                rgb_memory = rgb_memory + fusion_memory[:s] * attention_weights[:s]
                temperature_memory = temperature_memory + fusion_memory[s:] * attention_weights[s:]

            else:
                rgb_memory = rgb_memory
                temperature_memory = temperature_memory

        if gate_flag:
            output = fusion_memory[:s] * attention_weights[:s] + fusion_memory[s:] * attention_weights[s:]
        else:
            output = rgb_memory

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerCross(nn.Module):
    def __init__(self, cross_layer, encoder_layer, num_layers, cross_rgb_norm=None, cross_temperature_norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_layer, num_layers)
        self.rgb_encoder_layers = _get_clones(encoder_layer, num_layers)
        self.temperature_encoder_layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.cross_rgb_norm = cross_rgb_norm
        self.cross_temperature_norm = cross_temperature_norm
        self.return_intermediate = return_intermediate

    def forward(self, rgb_memory, temperature_memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                rgb_pos: Optional[Tensor] = None,
                temperature_pos: Optional[Tensor] = None):

        rgb_intermediate = []
        temperature_intermediate = []

        for i in range(len(self.layers)):
            rgb_memory, temperature_memory = self.layers[i](rgb_memory, temperature_memory, tgt_mask=tgt_mask,
                                                            memory_mask=memory_mask,
                                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                                            memory_key_padding_mask=memory_key_padding_mask,
                                                            rgb_pos=rgb_pos, temperature_pos=temperature_pos)

            rgb_memory = self.rgb_encoder_layers[i](rgb_memory, src_mask=memory_mask,
                                                    src_key_padding_mask=memory_key_padding_mask, pos=rgb_pos)

            temperature_memory = self.temperature_encoder_layers[i](temperature_memory, src_mask=memory_mask,
                                                                    src_key_padding_mask=memory_key_padding_mask,
                                                                    pos=temperature_pos)

        # for layer in self.layers:
        #     rgb_memory, temperature_memory = layer(rgb_memory, temperature_memory, tgt_mask=tgt_mask,
        #                                            memory_mask=memory_mask,
        #                                            tgt_key_padding_mask=tgt_key_padding_mask,
        #                                            memory_key_padding_mask=memory_key_padding_mask,
        #                                            rgb_pos=rgb_pos, temperature_pos=temperature_pos)
        # if self.return_intermediate:
        #     intermediate.append(self.norm(output))

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)
        #
        # if self.return_intermediate:
        #     return torch.stack(intermediate)

        if self.cross_rgb_norm is not None:
            rgb_memory = self.cross_rgb_norm(rgb_memory)

        if self.cross_temperature_norm is not None:
            temperature_memory = self.cross_temperature_norm(temperature_memory)

        return rgb_memory, temperature_memory


class TransformerCrossLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
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
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, rgb_memory, temperature_memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     rgb_pos: Optional[Tensor] = None,
                     temperature_pos: Optional[Tensor] = None):
        rgb_q = rgb_k = self.with_pos_embed(rgb_memory, rgb_pos)
        temperature_q = temperature_k = self.with_pos_embed(temperature_memory, temperature_pos)
        rgb_v = rgb_memory
        temperature_v = temperature_memory

        rgb_output = self.multihead_attn(query=rgb_q,
                                         key=temperature_k,
                                         value=temperature_v, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)[0]
        rgb_memory = rgb_memory + self.dropout2(rgb_output)
        rgb_memory = self.norm1(rgb_memory)

        rgb_output = self.linear2(self.dropout(self.activation(self.linear1(rgb_memory))))
        rgb_memory = rgb_memory + self.dropout3(rgb_output)
        rgb_memory = self.norm2(rgb_memory)

        temperature_output = self.multihead_attn(query=temperature_q,
                                                 key=rgb_k,
                                                 value=rgb_v, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)[0]
        temperature_memory = temperature_memory + self.dropout2(temperature_output)
        temperature_memory = self.norm3(temperature_memory)

        temperature_output = self.linear2(self.dropout(self.activation(self.linear1(temperature_memory))))
        temperature_memory = temperature_memory + self.dropout3(temperature_output)
        temperature_memory = self.norm4(temperature_memory)

        return rgb_memory, temperature_memory

    def forward_pre(self, rgb_memory, temperature_memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    rgb_pos: Optional[Tensor] = None,
                    temperature_pos: Optional[Tensor] = None):
        # tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt2 = self.norm2(tgt)

        rgb_output = self.norm1(rgb_memory)
        temperature_output = self.norm3(temperature_memory)

        rgb_q = rgb_k = self.with_pos_embed(rgb_output, rgb_pos)
        temperature_q = temperature_k = self.with_pos_embed(temperature_output, temperature_pos)

        rgb_output = self.multihead_attn(query=rgb_q,
                                         key=temperature_k,
                                         value=temperature_output, attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)[0]
        rgb_memory = rgb_memory + self.dropout2(rgb_output)
        rgb_memory = self.norm2(rgb_memory)

        rgb_output = self.linear2(self.dropout(self.activation(self.linear1(rgb_memory))))
        rgb_memory = rgb_memory + self.dropout3(rgb_output)

        temperature_output = self.multihead_attn(query=temperature_q,
                                                 key=rgb_k,
                                                 value=temperature_output, attn_mask=memory_mask,
                                                 key_padding_mask=memory_key_padding_mask)[0]
        temperature_memory = temperature_memory + self.dropout2(temperature_output)
        temperature_memory = self.norm4(temperature_memory)

        temperature_output = self.linear2(self.dropout(self.activation(self.linear1(temperature_memory))))
        temperature_memory = temperature_memory + self.dropout3(temperature_output)

        return rgb_memory, temperature_memory

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                rgb_pos: Optional[Tensor] = None,
                temperature_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, rgb_pos, temperature_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, rgb_pos, temperature_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
