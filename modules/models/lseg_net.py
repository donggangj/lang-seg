import math
import types
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os


class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


class Unflatten(nn.Module):
    def __init__(self, dim):
        super(Unflatten, self).__init__()
        self.dim = dim

    def forward(self, x, size: torch.Tensor):
        shape = torch.tensor(x.shape, device=x.device)
        new_shape = torch.cat((shape[:self.dim], size, shape[self.dim + 1:]))
        new_shape_tolist: List[int] = new_shape.tolist()
        return x.view(new_shape_tolist)


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class TextEncoder(nn.Module):
    def __init__(self, clip_module):
        super(TextEncoder, self).__init__()
        self.dtype = clip_module.dtype
        self.token_embedding = clip_module.token_embedding
        self.positional_embedding = clip_module.positional_embedding
        self.transformer = clip_module.transformer
        self.ln_final = clip_module.ln_final
        self.text_projection = clip_module.text_projection

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class LSeg(BaseModel):
    def __init__(
            self,
            head,
            features=256,
            backbone="clip_vitl16_384",
            readout="project",
            channels_last=False,
            use_bn=False,
            **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }
        self.vision_hook_ids = hooks[backbone]

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=self.vision_hook_ids,
            use_readout=readout,
        )

        self.text_encoder = None

        self.unflatten = Unflatten(dim=2)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.scratch.output_conv = head

        self.text = clip.tokenize(self.labels)

    def init_after_loading(self):
        self.init_act_postprocessing()
        self.text_encoder = torch.jit.trace(TextEncoder(self.clip_pretrained), self.text.cuda())
        clip_pretrained = self.clip_pretrained
        self.clip_pretrained = None
        del clip_pretrained

    def init_act_postprocessing(self):
        act_postprocessing = (self.pretrained.act_postprocess1[:2],
                              self.pretrained.act_postprocess2[:2],
                              self.pretrained.act_postprocess3[:2],
                              self.pretrained.act_postprocess4[:2],
                              self.pretrained.act_postprocess1[3:],
                              self.pretrained.act_postprocess2[3:],
                              self.pretrained.act_postprocess3[3:],
                              self.pretrained.act_postprocess4[3:])
        for i in range(len(act_postprocessing)):
            setattr(self, f'act_postprocessing{i + 1}', act_postprocessing[i])

    def forward(self, x, tokens=torch.tensor([])):
        device = x.device
        if tokens.numel() == 0:
            text = self.text
        else:
            text = tokens

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = self.forward_vit_custom(x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3.forward_with_skip(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2.forward_with_skip(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1.forward_with_skip(path_2, layer_1_rn)

        text = text.to(device)
        logit_scale = self.logit_scale.to(device)
        text_features = self.text_encoder(text)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        logits_per_image = logit_scale * image_features.half() @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        assert self.arch_option not in [1, 2]
        # if self.arch_option in [1, 2]:
        #     for _ in range(self.block_depth - 1):
        #         out = self.scratch.head_block(out)
        #     out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)

        return out

    def forward_vit_custom(self, x):
        model = self.pretrained.model
        b, c, h, w = x.shape

        # encoder
        layer_1, layer_2, layer_3, layer_4 = self.forward_flex_custom(x)

        layer_1 = self.act_postprocessing1(layer_1)
        layer_2 = self.act_postprocessing2(layer_2)
        layer_3 = self.act_postprocessing3(layer_3)
        layer_4 = self.act_postprocessing4(layer_4)

        unflattened_size = torch.tensor([h // model.patch_size[1],
                                         w // model.patch_size[0]],
                                        device=x.device)
        layer_1 = self.unflatten(layer_1, unflattened_size)
        layer_2 = self.unflatten(layer_2, unflattened_size)
        layer_3 = self.unflatten(layer_3, unflattened_size)
        layer_4 = self.unflatten(layer_4, unflattened_size)

        layer_1 = self.act_postprocessing5(layer_1)
        layer_2 = self.act_postprocessing6(layer_2)
        layer_3 = self.act_postprocessing7(layer_3)
        layer_4 = self.act_postprocessing8(layer_4)

        return layer_1, layer_2, layer_3, layer_4

    def forward_vit(self, x):
        model = self.pretrained.model
        b, c, h, w = x.shape

        # encoder
        layer_1, layer_2, layer_3, layer_4 = self.forward_flex(x)

        layer_1 = self.act_postprocessing1(layer_1)
        layer_2 = self.act_postprocessing2(layer_2)
        layer_3 = self.act_postprocessing3(layer_3)
        layer_4 = self.act_postprocessing4(layer_4)

        unflattened_size = torch.tensor([h // model.patch_size[1],
                                         w // model.patch_size[0]],
                                        device=x.device)
        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1, unflattened_size)
        if layer_2.ndim == 3:
            layer_2 = self.unflatten(layer_2, unflattened_size)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3, unflattened_size)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4, unflattened_size)

        layer_1 = self.act_postprocessing5(layer_1)
        layer_2 = self.act_postprocessing6(layer_2)
        layer_3 = self.act_postprocessing7(layer_3)
        layer_4 = self.act_postprocessing8(layer_4)

        return layer_1, layer_2, layer_3, layer_4

    def forward_flex_custom(self, x):
        vision_hook_ids = self.vision_hook_ids
        model = self.pretrained.model
        layers = []
        b, c, h, w = x.shape

        pos_embed = self._resize_pos_embed(model.pos_embed, h // model.patch_size[1], w // model.patch_size[0])

        B = x.shape[0]

        x = model.patch_embed.proj(x).flatten(2).transpose(1, 2)

        cls_tokens = model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = model.pos_drop(x)

        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i in vision_hook_ids:
                layers.append(x)

        x = model.norm(x)

        return layers

    def forward_flex(self, x):
        vision_hook_ids = self.vision_hook_ids
        model = self.pretrained.model
        layers = []
        b, c, h, w = x.shape

        pos_embed = self._resize_pos_embed(
            model.pos_embed, h // model.patch_size[1], w // model.patch_size[0]
        )

        B = x.shape[0]

        if hasattr(model.patch_embed, "backbone"):
            x = model.patch_embed.backbone(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = model.patch_embed.proj(x).flatten(2).transpose(1, 2)

        if getattr(model, "dist_token", None) is not None:
            cls_tokens = model.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            dist_token = model.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = model.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + pos_embed
        x = model.pos_drop(x)

        for i, blk in enumerate(model.blocks):
            x = blk(x)
            if i in vision_hook_ids:
                layers.append(x)

        x = model.norm(x)

        return layers

    def _resize_pos_embed(self, posemb, gs_h: int, gs_w: int):
        model = self.pretrained.model
        posemb_tok = posemb[:, :model.start_index]
        posemb_grid = posemb[0, model.start_index:]
        gs_old = torch.sqrt(torch.tensor(posemb_grid.shape[0], device=posemb.device)).long()

        posemb_grid = posemb_grid.reshape(1, int(gs_old), int(gs_old), -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb


class LSegNet(LSeg):
    """Network for semantic segmentation."""

    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)
