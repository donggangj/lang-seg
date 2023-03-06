import time

import numpy as np
import torch
from PIL import Image
from matplotlib import patches as mpatches, pyplot as plt
from openvino.runtime import Core


def get_time_stamp(fmt: str = '%y-%m-%d-%H-%M-%SZ'):
    return time.strftime(fmt, time.gmtime())


def get_physical_device_name(device: str):
    device_name = 'unknown'
    if device.lower().startswith('cpu'):
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'model name' in line:
                device_name = line.split(':')[-1].strip()
                break
    elif device.lower().startswith('cuda') and 'cuda' in torch.__dict__ and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
    elif device.lower().startswith('hpu') and 'hpu' in torch.__dict__ and torch.hpu.is_available():
        device_name = torch.hpu.get_device_name()
    elif device.lower().startswith('gpu'):
        core = Core()
        available_gpus = [_device for _device in core.available_devices if 'gpu' in _device.lower()]
        if device in available_gpus:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
    return device_name


def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete


def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)

    patches = []
    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0,
                         new_palette[index * 3 + 2] / 255.0]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


def show_result(image, predict, labels, alpha, save_path, title=''):
    # show results
    new_palette = get_new_pallete(len(labels))
    mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=labels)
    img = image[0].permute(1, 2, 0)
    img = img * 0.5 + 0.5
    img = Image.fromarray(np.uint8(255 * img)).convert("RGBA")
    seg = mask.convert("RGBA")
    out = Image.blend(img, seg, alpha)
    fig = plt.figure(figsize=(19.2, 3.6))
    axes = fig.subplots(1, 3)
    axes[0].imshow(img)
    axes[0].xaxis.set_ticks([])
    axes[0].yaxis.set_ticks([])
    axes[0].set_xlabel('Original')
    axes[1].imshow(out)
    axes[1].xaxis.set_ticks([])
    axes[1].yaxis.set_ticks([])
    axes[1].set_title(title)
    axes[1].set_xlabel('Original + Predicted Mask')
    axes[2].imshow(seg)
    axes[2].xaxis.set_ticks([])
    axes[2].yaxis.set_ticks([])
    axes[2].set_xlabel('Predicted Mask')
    axes[2].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
    fig.savefig(save_path)
