from re import I
import torch
import torch.nn as nn
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .transformer import TransformerDecoderLayer, TransformerDecoder
from .positional_encoding import build_position_encoding
import pdb

class SpatioTempEncoder(nn.Module):
    def __init__(self, args):
        super(SpatioTempEncoder, self).__init__()
        self.num_encoder_layers = args.enc_layers
        self.num_decoder_layers = args.dec_layers
        self.activation = "gelu"
        self.normalize_before = args.pre_norm
        self.seq_len = args.sequence_length

        encoder_layer = TransformerEncoderLayer(args.d_model, args.nheads, args.dim_feedforward,
                                                args.tf_dropout, self.activation, self.normalize_before)
        encoder_norm = nn.LayerNorm(args.d_model) if self.normalize_before else None
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)

        self.input_proj = nn.Conv2d(args.dim_feedforward, args.d_model, kernel_size=1)
        
        self.pos_encoder = build_position_encoding(args)
    
    def forward(self, src):
        pos_embed = self.pos_encoder(src)  
        bsz = pos_embed.shape[0]
        device = src.device

        src = self.input_proj(src)  
        _, c, h, w = src.shape  
        src = src.flatten(2).reshape(bsz, self.seq_len, c, h*w)  
        src = src.permute(1, 3, 0, 2).flatten(start_dim=0, end_dim=1) 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  
        mask = torch.zeros([bsz, self.seq_len*h*w], dtype=bool).to(device)
        hs = self.transformer_encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # 1+8*49, bsz, c

        return hs, mask, pos_embed