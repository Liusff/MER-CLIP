# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.fileio import load
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import _load_checkpoint_with_prefix
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmaction.utils import ConfigType, get_str_type
from .base import BaseHead

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, ff_dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    Self-attention module. 
    Currently supports both full self-attention on all the input tokens,
    or only-spatial/only-temporal self-attention. 

    See Anurag Arnab et al.
    ViVIT: A Video Vision Transformer.
    and 
    Gedas Bertasius, Heng Wang, Lorenzo Torresani.
    Is Space-Time Attention All You Need for Video Understanding?

    Modified from 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(
        self,
        dim,
        num_heads=12,
        attn_dropout=0.,
        ff_dropout=0.,
        einops_from=None,
        einops_to=None,
        **einops_dims,
    ):
        super().__init__()
        self.num_heads = num_heads
        dim_head = dim // num_heads
        self.scale = dim_head ** -0.5

        self.to_qkv         = nn.Linear(dim, dim * 3)
        self.attn_dropout   = nn.Dropout(attn_dropout)
        self.proj           = nn.Linear(dim, dim)
        self.ff_dropout     = nn.Dropout(ff_dropout)

        if einops_from is not None and einops_to is not None:
            self.partial = True
            self.einops_from = einops_from
            self.einops_to = einops_to
            self.einops_dims = einops_dims
        else:
            self.partial = False

    def forward(self, x):
        if self.partial:
            return self.forward_partial(
                x,
                self.einops_from,
                self.einops_to,
                **self.einops_dims,
            )
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # import os
        # for i in range(12):
        #     if not os.path.exists(f"./debug/transformer_visualization/layer_{i}.pyth"):
        #         break
        # torch.save(attn,f"./debug/transformer_visualization/layer_{i}.pyth")
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

    def forward_partial(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q *= self.scale

        # splice out classification token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let classification token attend to key / values of all patches across time and space
        cls_attn = (cls_q @ k.transpose(1,2)).softmax(-1)
        cls_attn = self.attn_dropout(cls_attn)
        cls_out = cls_attn @ v

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b () d -> (b r) () d', r = r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim = 1)
        v_ = torch.cat((cls_v, v_), dim = 1)

        # attention
        attn = (q_ @ k_.transpose(1, 2)).softmax(-1)
        attn = self.attn_dropout(attn)
        x = attn @ v_

        # merge back time or space
        x = rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim = 1)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        x = self.proj(x)
        x = self.ff_dropout(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py.
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class BaseTransformerLayer(nn.Module):
    def __init__(self, dim_override=None, num_heads_override=None, attn_dropout_override=None,
                 ff_dropout_override=None, mlp_mult_override=None, drop_path_rate=0.0):
        """
        Args: 
            drop_path_rate  (float): rate for drop path. 
                See models/base/base_blocks.py L897-928.
        """
        super().__init__()

        dim             = dim_override
        num_heads       = num_heads_override
        attn_dropout    = attn_dropout_override
        ff_dropout      = ff_dropout_override
        mlp_mult        = mlp_mult_override
        drop_path       = drop_path_rate

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_dropout=attn_dropout, ff_dropout=ff_dropout
        )
        self.norm_ffn = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = FeedForward(dim=dim, mult=mlp_mult, ff_dropout=ff_dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


@MODELS.register_module()
class TransfClsHead_Vit(BaseHead):
    """Classification head for UniFormer. supports loading pretrained
    Kinetics-710 checkpoint to fine-tuning on other Kinetics dataset.

    A pytorch implement of: `UniFormerV2: Spatiotemporal
    Learning by Arming Image ViTs with Video UniFormer
    <https://arxiv.org/abs/2211.09552>`

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict or ConfigDict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        dropout_ratio (float): Probability of dropout layer.
            Defaults to : 0.0.
        channel_map (str, optional): Channel map file to selecting
            channels from pretrained head with extra channels.
            Defaults to None.
        init_cfg (dict or ConfigDict, optional): Config to control the
           initialization. Defaults to
           ``[
            dict(type='TruncNormal', layer='Linear', std=0.01)
           ]``.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 drop_path: float = 0.0,
                 depth: int = 2,
                 num_heads: int = 4,
                 attn_drop_out: float = 0.0,
                 ffn_drop_out: float = 0.0,
                 mlp_mult: int = 4,
                 multi_layer_forward: bool = True,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 dropout_ratio: float = 0.0,
                 channel_map: Optional[str] = None,
                 init_cfg: Optional[dict] = dict(
                     type='TruncNormal', layer='Linear', std=0.02),
                 **kwargs) -> None:
        super().__init__(
            num_classes, in_channels, loss_cls, init_cfg=init_cfg, **kwargs)
        self.channel_map = channel_map
        self.dropout_ratio = dropout_ratio
        self.multi_layer_forward = multi_layer_forward
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            BaseTransformerLayer(num_heads_override=num_heads, dim_override=in_channels, mlp_mult_override=mlp_mult,
                                                            drop_path_rate=dpr[i], attn_dropout_override=attn_drop_out, ff_dropout_override=ffn_drop_out)
            for i in range(depth)])
        self.norm =  nn.LayerNorm(in_channels, eps=1e-6)
        nn.init.constant_(self.norm.bias, 0.0)
        nn.init.constant_(self.norm.weight, 1.0)

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)


    def _select_channels(self, stact_dict):
        selected_channels = load(self.channel_map)
        for key in stact_dict:
            stact_dict[key] = stact_dict[key][selected_channels]

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        if get_str_type(self.init_cfg['type']) == 'Pretrained':
            assert self.channel_map is not None, \
                'load cls_head weights needs to specify the channel map file'
            logger = MMLogger.get_current_instance()
            pretrained = self.init_cfg['checkpoint']
            logger.info(f'load pretrained model from {pretrained}')
            state_dict = _load_checkpoint_with_prefix(
                'cls_head.', pretrained, map_location='cpu')
            self._select_channels(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(msg)
        else:
            super().init_weights()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The classification scores for input samples.
        """
        if self.multi_layer_forward:
            cls_score, feat = self.forward_multi_layer(x)
        else:
            cls_score, feat = self.forward_single_layer(x)
        '''
        # [N, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        '''
        return cls_score
    
    def forward_single_layer(self, x):
        out = x
        for blk in self.blocks:
            out = blk(out)
        feat = out
        out = self.norm(out.mean(dim=1))
        if hasattr(self, "dropout"):
            out = self.dropout(out)
        out = self.fc_cls(out)
        return out, feat

    def forward_multi_layer(self, x):
        out = x
        out_list = []
        for blk in self.blocks:
            out = blk(out)
            out_list.append(out)
        feat = out
        out_full = torch.stack(out_list, dim=1).flatten(0, 1)
        out_full = self.norm(out_full.mean(dim=1))
        #print("out_full: ", out_full.shape)
        #out_full = self.norm(out_full[:, 0, :])
        if hasattr(self, "dropout"):
            out_full = self.dropout(out_full)
        out_full = self.fc_cls(out_full)
        out_full = out_full.view(x.size(0), len(out_list), -1).mean(dim=1)
        return out_full, feat
