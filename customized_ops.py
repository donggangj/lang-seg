from collections import OrderedDict
from typing import Tuple, Union, Optional

import torch
from torch import nn, Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
import torch.nn.functional as F
"""
Add import to clip/model.py
=====================================================
from typing import  Optional
from torch import Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter
=====================================================
"""


class LayerNorm(nn.LayerNorm):
    """
    From clip/model.py.

    ORIGINAL CODE:
    =====================================================

    class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

    =====================================================


    Subclass torch's LayerNorm to handle fp16.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = F.layer_norm(x.type(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return ret.type(orig_type)


class CLIP(nn.Module):
    """
    From clip/model.py

    ORIGINAL CODE:
    =====================================================
    def __init__(...):
        ...
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        ...

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
    =====================================================
    """

    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        # Replace corresponding part only
        self.transformer = TransformerCustom(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.in_proj_weight.dtype

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(p=2, dim=1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttentionCustom(nn.Module):
    """
    Add to clip/model.py
    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads) -> None:
        factory_kwargs = {'device': None, 'dtype': None}
        super(MultiheadAttentionCustom, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self._qkv_same_embed_dim = True

        self.num_heads = num_heads
        self.dropout = 0.
        self.batch_first = False
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True, **factory_kwargs)

        self.bias_k = self.bias_v = None

        self.add_zero_attn = False

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttentionCustom, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor) -> Tuple[Tensor, Optional[Tensor]]:

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=False,
            key_padding_mask=None, need_weights=False,
            attn_mask=attn_mask, average_attn_weights=True)

        return attn_output, attn_output_weights


class ResidualAttentionBlockCustom(nn.Module):
    """
    Add to clip/model.py
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor):
        super().__init__()

        self.attn = MultiheadAttentionCustom(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, attn_mask=self.attn_mask.to(dtype=x.dtype, device=x.device))[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerCustom(nn.Module):
    """
    Add to clip/model.py
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockCustom(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


def convert_weights(model: nn.Module):
    """
    From clip/model.py

    ORIGINAL CODE:
    =====================================================
    def convert_weights(model: nn.Module):

        def _convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

            if isinstance(l, nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        model.apply(_convert_weights_to_fp16)
    =====================================================

    Convert applicable model parameters to fp16
    """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if type(l) in (nn.MultiheadAttention, MultiheadAttentionCustom):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
