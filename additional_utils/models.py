###########################################################################
# Referred to: https://github.com/zhanghang1989/PyTorch-Encoding
###########################################################################
import math

import clip
import numpy as np
from typing import Any, Union

from torch import Tensor
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter
import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['LSeg_MultiEvalModule',
           'LSegMultiEvalAlter',
           ]


class LSeg_MultiEvalModule(DataParallel):
    """Multi-size Segmentation Eavluator"""

    def __init__(self, module, device_ids=None, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        super(LSeg_MultiEvalModule, self).__init__(module, device_ids)
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.scales = scales
        self.flip = flip
        print('MultiEvalModule: base_size {}, crop_size {}'. \
              format(self.base_size, self.crop_size))

    def parallel_forward(self, inputs, label_set='', **kwargs):
        """Multi-GPU Mult-size Evaluation

        Args:
            inputs: list of Tensors
        """
        if len(label_set) < 10:
            print('** MultiEvalModule parallel_forward phase: {} **'.format(label_set))
        self.nclass = len(label_set)
        if type(label_set) in [str, list]:
            label_set = clip.tokenize(label_set)
        inputs = [(input.unsqueeze(0).cuda(device),)
                  for input, device in zip(inputs, self.device_ids)]
        replicas = self.replicate(self, self.device_ids[:len(inputs)])
        kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        outputs = parallel_apply(replicas, inputs, label_set, kwargs)
        return outputs

    def forward(self, image, label_set=''):
        """Mult-size Evaluation"""
        # only single image is supported for evaluation
        if len(label_set) < 10:
            print('** MultiEvalModule forward phase: {} **'.format(label_set))
        batch, _, h, w = image.size()
        assert (batch == 1)
        self.nclass = len(label_set)
        stride_rate = 2.0 / 3.0
        crop_size = self.crop_size
        stride = int(crop_size * stride_rate)
        with torch.cuda.device_of(image):
            scores = image.new().resize_(batch, self.nclass, h, w).zero_().cuda()

        for scale in self.scales:
            long_size = int(math.ceil(self.base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            """
            short_size = int(math.ceil(self.base_size * scale))
            if h > w:
                width = short_size
                height = int(1.0 * h * short_size / w)
                long_size = height
            else:
                height = short_size
                width = int(1.0 * w * short_size / h)
                long_size = width
            """
            # resize image to current size
            cur_img = resize_image(image, height, width, **self.module._up_kwargs)
            if long_size <= crop_size:
                pad_img = pad_image(cur_img, self.module.mean,
                                    self.module.std, crop_size)
                outputs = module_inference(self.module, pad_img, label_set, self.flip)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.module.mean,
                                        self.module.std, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.shape  # .size()
                assert (ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                with torch.cuda.device_of(image):
                    outputs = image.new().resize_(batch, self.nclass, ph, pw).zero_().cuda()
                    count_norm = image.new().resize_(batch, 1, ph, pw).zero_().cuda()
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.module.mean,
                                                 self.module.std, crop_size)
                        output = module_inference(self.module, pad_crop_img, label_set, self.flip)
                        outputs[:, :, h0:h1, w0:w1] += crop_image(output,
                                                                  0, h1 - h0, 0, w1 - w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert ((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]
            score = resize_image(outputs, h, w, **self.module._up_kwargs)
            scores += score
        return scores


def module_inference(module, image, label_set, flip=True):
    output = module.net(image, label_set)
    if flip:
        fimg = flip_image(image)
        foutput = module.net(fimg, label_set)
        output += flip_image(foutput)
    return output


def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)


def pad_image(img, mean, std, crop_size):
    b, c, h, w = img.shape  # .size()
    assert (c == 3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    pad_values = -np.array(mean) / np.array(std)
    img_pad = img.new().resize_(b, c, h + padh, w + padw)
    for i in range(c):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, padw, 0, padh), value=pad_values[i])
    assert (img_pad.size(2) >= crop_size and img_pad.size(3) >= crop_size)
    return img_pad


def crop_image(img, h0: int, h1: int, w0: int, w1: int):
    return img[:, :, h0:h1, w0:w1]


def flip_image(img):
    assert (img.dim() == 4)
    idx = torch.arange(img.size(3) - 1, -1, -1).type_as(img).long().to(img.device)
    return img.index_select(3, idx)


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, label_set, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    # out = modules[0](*inputs[0], label_set)
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(i, module, input, label_set, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, label_set, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, label_set, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], label_set, kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


@torch.jit.script
def get_shape(x: Tensor):
    return torch.tensor(x.shape, device=x.device)


@torch.jit.script
def pad_image_script(img, mean, std, crop_size):
    shape = get_shape(img)
    b, c, h, w = shape[0], shape[1], shape[2], shape[3]
    device = img.device
    padh = crop_size - h if h < crop_size else torch.tensor(0, device=device)
    padw = crop_size - w if w < crop_size else torch.tensor(0, device=device)
    pad_values = -mean / std
    img_pad = torch.zeros(b, c, h + padh, w + padw).to(device)
    for i in range(int(c)):
        # note that pytorch pad params is in reversed orders
        img_pad[:, i, :, :] = F.pad(img[:, i, :, :], (0, int(padw), 0, int(padh)), value=pad_values[i])
    return img_pad
#
#
# def module_inference_script(module, image: Tensor, label_set, flip=True):
#     output = module.evaluate_random(image, label_set)
#     if flip:
#         fimg = flip_image(image)
#         foutput = module.evaluate_random(fimg, label_set)
#         output += flip_image(foutput)
#     return output
#
#
# @torch.jit.script
# def grid_eval(net, pad_img: Tensor, label_set: Tensor, h_grids: Tensor, w_grids: Tensor,
#               n_class: Tensor, crop_size: Tensor, stride: Tensor,
#               mean: Tensor, std: Tensor, flip: bool):
#     pad_shape = get_shape(pad_img)
#     batch, ph, pw = pad_shape[0], pad_shape[2], pad_shape[3]  # .size()
#     outputs = torch.zeros(batch, n_class, ph, pw).cuda()
#     count_norm = torch.zeros(batch, 1, ph, pw).cuda()
#     # grid evaluation
#     for idh in range(int(h_grids)):
#         for idw in range(int(w_grids)):
#             h0 = torch.tensor(idh * stride, dtype=torch.int32)
#             w0 = torch.tensor(idw * stride, dtype=torch.int32)
#             h1 = torch.min(h0 + crop_size, ph)
#             w1 = torch.min(w0 + crop_size, pw)
#             crop_img = crop_image(pad_img, h0, h1, w0, w1)
#             # pad if needed
#             pad_crop_img = pad_image_script(crop_img, mean,
#                                             std, crop_size)
#             output = module_inference_script(net, pad_crop_img, label_set, flip)
#             outputs[:, :, h0:h1, w0:w1] += crop_image(output,
#                                                       0, h1 - h0, 0, w1 - w0)
#             count_norm[:, :, h0:h1, w0:w1] += 1
#     return outputs, count_norm
#
#
# @torch.jit.script
# def loop_scale(net, image: Tensor, label_set: Tensor,
#                scales: Tensor, n_class: Tensor, base_size: Tensor, crop_size: Tensor, stride: Tensor,
#                mean: Tensor, std: Tensor, flip: bool):
#     shape = get_shape(image)
#     batch, h, w = shape[0], shape[2], shape[3]
#     scores = torch.zeros(batch, n_class, h, w).cuda()
#     for scale in scales:
#         long_size = torch.ceil(torch.tensor(base_size * scale)).to(torch.int32)
#         if h > w:
#             height = long_size
#             width = torch.floor(torch.tensor(1.0 * w * long_size / h + 0.5)).to(torch.int32)
#             short_size = width
#         else:
#             width = long_size
#             height = torch.floor(torch.tensor(1.0 * h * long_size / w + 0.5)).to(torch.int32)
#             short_size = height
#         # resize image to current size
#         cur_img = InterpolateCompatible.apply(image, torch.tensor([height, width]))
#         if long_size <= crop_size:
#             pad_img = pad_image_script(cur_img, mean,
#                                        std, crop_size)
#             outputs = module_inference_script(net, pad_img, label_set, flip)
#             outputs = crop_image(outputs, 0, height, 0, width)
#         else:
#             if short_size < crop_size:
#                 # pad if needed
#                 pad_img = pad_image_script(cur_img, mean,
#                                            std, crop_size)
#             else:
#                 pad_img = cur_img
#             shape = get_shape(pad_img)
#             ph, pw = shape[2], shape[3]  # .size()
#             # grid forward and normalize
#             h_grids = torch.tensor(torch.ceil(1.0 * (ph - crop_size) / stride) + 1, dtype=torch.int32)
#             w_grids = torch.tensor(torch.ceil(1.0 * (pw - crop_size) / stride) + 1, dtype=torch.int32)
#             outputs, count_norm = grid_eval(net, pad_img, label_set,
#                                             h_grids, w_grids, n_class, crop_size, stride, mean, std, flip)
#             outputs = outputs / count_norm
#             outputs = outputs[:, :, :height, :width]
#
#         score = InterpolateCompatible.apply(outputs, torch.tensor([h, w]))
#         scores += score
#         return scores


class InterpolateCompatible(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) >= 4:
            t, scales, mode, align_corners = args[:4]
            return F.interpolate(t,
                                 scales[-2:].tolist(),
                                 mode=mode,
                                 align_corners=align_corners)
        else:
            t, scales = args[:2]
            return F.interpolate(t,
                                 scales[-2:].tolist(),
                                 mode='bilinear',
                                 align_corners=True)

    @staticmethod
    def symbolic(g, t, scales, mode='linear', align_corners=True):
        return g.op('Resize',
                    t,
                    g.op('Constant', value_t=torch.tensor([], dtype=torch.float32)),
                    scales,
                    coordinate_transformation_mode_s='align_corners' if align_corners else 'pytorch_half_pixel',
                    mode_s=mode)


class LSegMultiEvalAlter(torch.nn.Module):
    def __init__(self, net, device_ids=None, flip=True,
                 scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
                 n_class=1,
                 sample_input=()):
        super().__init__()
        self.base_size = net.base_size
        self.crop_size = net.crop_size
        self.nclass = n_class
        self.net = torch.jit.script(net.net)
        self.device_ids = device_ids
        self.scales = list(scales)
        self.flip = flip
        self.mean = torch.tensor(net.mean)
        self.std = torch.tensor(net.std)
        print('MultiEvalModule: base_size {}, crop_size {}'.format(self.base_size, self.crop_size))

    def forward(self, image, tokens=torch.tensor([])):
        """Multi-size Evaluation"""
        # only single image is supported for evaluation
        n_class = get_shape(tokens)[0]
        if n_class > 0:
            self.nclass = n_class
        stride_rate = 2.0 / 3.0
        stride = torch.tensor(self.crop_size * stride_rate, dtype=torch.int32).to(image.device)

        return self.loop_scale(image, tokens, stride)

    def net_forward(self, image: Tensor, label_set):
        output = self.net(image, label_set)
        if self.flip:
            fimg = flip_image(image)
            foutput = self.net(fimg, label_set)
            output += flip_image(foutput)
        return output

    def grid_eval(self, pad_img: Tensor, label_set: Tensor, h_grids: Tensor, w_grids: Tensor,
                  crop_size: Tensor, stride: Tensor):
        pad_shape = get_shape(pad_img)
        batch, ph, pw = pad_shape[0], pad_shape[2], pad_shape[3]  # .size()
        outputs = torch.zeros(batch, self.nclass, ph, pw).to(pad_img.device)
        count_norm = torch.zeros(batch, 1, ph, pw).to(pad_img.device)
        # grid evaluation
        for idh in range(int(h_grids)):
            for idw in range(int(w_grids)):
                h0 = (idh * stride).to(torch.int32)
                w0 = (idw * stride).to(torch.int32)
                h1 = torch.min(h0 + crop_size, ph)
                w1 = torch.min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image_script(crop_img, self.mean,
                                                self.std, crop_size)
                output = self.net_forward(pad_crop_img, label_set)
                outputs[:, :, h0:h1, w0:w1] += crop_image(output,
                                                          0, h1 - h0, 0, w1 - w0)
                count_norm[:, :, h0:h1, w0:w1] += 1
        return outputs, count_norm

    def loop_scale(self, image: Tensor, label_set: Tensor, stride: Tensor):
        base_size = torch.tensor(self.base_size).to(image.device)
        crop_size = torch.tensor(self.crop_size).to(image.device)
        shape = get_shape(image)
        batch, h, w = shape[0], shape[2], shape[3]
        scores = torch.zeros(batch, self.nclass, h, w).to(image.device)
        for scale in torch.tensor(self.scales).to(image.device):
            long_size = torch.ceil(base_size * scale).to(torch.int32)
            if h > w:
                height = long_size
                width = torch.floor(1.0 * w * long_size / h + 0.5).to(torch.int32)
                short_size = width
            else:
                width = long_size
                height = torch.floor(1.0 * h * long_size / w + 0.5).to(torch.int32)
                short_size = height
            # resize image to current size
            cur_img = F.interpolate(image, (int(height), int(width)), mode='bilinear', align_corners=True)
            if long_size <= crop_size:
                pad_img = pad_image_script(cur_img, self.mean,
                                           self.std, crop_size)
                outputs = self.net_forward(pad_img, label_set)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = pad_image_script(cur_img, self.mean,
                                               self.std, crop_size)
                else:
                    pad_img = cur_img
                shape = get_shape(pad_img)
                ph, pw = shape[2], shape[3]  # .size()
                # grid forward and normalize
                h_grids = torch.ceil(1.0 * (ph - crop_size) / stride).to(torch.int32) + 1
                w_grids = torch.ceil(1.0 * (pw - crop_size) / stride).to(torch.int32) + 1
                outputs, count_norm = self.grid_eval(pad_img, label_set, h_grids, w_grids, crop_size, stride)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            score = F.interpolate(outputs, (int(h), int(w)), mode='bilinear', align_corners=True)
            scores += score
        return scores
