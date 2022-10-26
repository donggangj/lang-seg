import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
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

    def __init__(self, *args, **kwargs):
        super(LayerNorm, self).__init__()

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = F.layer_norm(x.type(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return ret.type(orig_type)


class CLIP(nn.Module):
    """
    From clip/model.py

    ORIGINAL CODE:
    =====================================================
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
