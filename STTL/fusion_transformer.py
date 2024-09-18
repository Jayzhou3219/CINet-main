import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from STTL.transformer_encoder import TransformerEncoderLayer, TransformerEncoder

class TransformerModel(nn.Module):

    def __init__(self, additional_info_upper, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(additional_info_upper, ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):  #
        # n_mask = mask + torch.eye(mask.shape[0], mask.shape[0]).cuda()
        # n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        output = self.transformer_encoder(src, mask=False)

        return output
