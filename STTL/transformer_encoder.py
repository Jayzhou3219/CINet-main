import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from STTL.multiheadAttention import MultiheadAttention


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        # self.layers = _get_clones(encoder_layer, num_layers)
        self.layers = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):

        output = src

        atts = []

        for i in range(self.num_layers):
            output, attn = self.layers(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, additional_info_upper, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu"):  # d_model:编码向量长度
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(additional_info_upper, d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(64, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, 64)

        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        self_attn = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)  # self.self_attn=MultiheadAttention(d_model, nhead, dropout=dropout)
        src2, attn = self_attn(src)
        # 第一个残差连接
        src_add = src + self.dropout1(src2)  # (8,32,64)
        # 追加layer norm
        # src = self.norm1(src)
        src_add = self.norm1(src_add[-1])
        # 全连接层
        if hasattr(self, "activation"):  # hasattr(object,name):判断对象object是否包含名为name的特性
            src3 = self.linear2(self.dropout(self.activation(self.linear1(src_add))))
        else:  # for backward compatibility
            src3 = self.linear2(self.dropout(F.relu(self.linear1(src_add))))
        # 第二个残差连接
        src_add_2 = src_add + self.dropout2(src3)
        # 追加layer norm
        # src = self.norm2(src)self.dropout2(src3)
        return src_add_2, attn