
from typing import Tuple, Optional

import einops
import torch
from mmengine.model import BaseModule
from mmcv.cnn.bricks.transformer import FFN

from mmpretrain.models import build_norm_layer
from mmpretrain.models.backbones.vit_sam import Attention, window_partition, window_unpartition

from opencd.registry import MODELS

import torch
import torch.nn as nn
import torch.nn.functional as F

@MODELS.register_module()
class TimeFusionTransformerEncoderLayer(BaseModule):
    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 feedforward_channels: int,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 num_fcs: int = 2,
                 qkv_bias: bool = True,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN'),
                 use_rel_pos: bool = False,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.window_size = window_size

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        if self.window_size > 0:
            in_channels = embed_dims * 2
            self.down_channel = torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, bias=False)
            self.down_channel.weight.data.fill_(1.0 / in_channels)

            self.soft_ffn = torch.nn.Sequential(
                torch.nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(embed_dims, embed_dims, kernel_size=1, stride=1),
            )

        self.crossAtt = CrossAttention(channels = embed_dims, num_heads=8)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x

        x = self.ffn(self.ln2(x), identity=x)

        if self.window_size > 0:
            x = einops.rearrange(x, 'b h w d -> b d h w')
            x0 = x[:x.size(0)//2]
            x1 = x[x.size(0)//2:]

            x0_1 = torch.cat([x0, x1], dim=1)

            activate_map = self.down_channel(x0_1)

            activate_map = torch.sigmoid(activate_map)

            x0 = x0 + self.soft_ffn(x1 * activate_map)

            x = torch.cat([x0, x1], dim=0)

            x = einops.rearrange(x, 'b d h w -> b h w d')
        return x

class CrossAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert self.head_dim * num_heads == channels, "通道数必须能被头数整除"

        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)

        self.fc_out = nn.Linear(channels, channels)

    def forward(self, x0, x1):

        batch_size, C, H, W = x0.shape
        N = H * W

        x0_flat = x0.permute(0, 2, 3, 1).reshape(batch_size, N, C)
        x1_flat = x1.permute(0, 2, 3, 1).reshape(batch_size, N, C)

        q = self.w_q(x0_flat)
        k = self.w_k(x1_flat)
        v = self.w_v(x1_flat)

        q = q.reshape(batch_size, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        att_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(att_weights, v)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, N, C)
        out = self.fc_out(out)

        enhanced_x0 = out.reshape(batch_size, H, W, C).permute(0, 3, 1, 2)

        return enhanced_x0
