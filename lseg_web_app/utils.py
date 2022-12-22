import argparse
import hashlib
import json
import time
from os import makedirs
from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Language-driven Semantic Segmentation on Intel Habana")

        # app option
        parser.add_argument(
            "--config_path",
            type=str,
            default=".app/lseg_habana_app_config.json",
            help="path to the app config",
        )

        self.parser = parser

    def parse(self):
        opt = self.parser.parse_args([])
        return opt


class MD5Table:
    def __init__(self, md5_table_path: str):
        self._path = md5_table_path
        self._md5 = {}
        self.load_md5_table()

    def load_md5_table(self):
        try:
            with open(self._path, 'r') as f:
                self._md5 = json.load(f)
        except Exception as err:
            print(f"Invalid md5 table: {err}")
            self._md5 = {}
        return self._md5.copy()

    def save_md5_table(self):
        try:
            with open(self._path, 'w') as f:
                json.dump(self._md5, f, indent=4)
        except Exception as err:
            print(err)

    def update_md5_table(self, content_path: str):
        md5 = MD5Table.calc_file_md5(content_path)
        if md5 not in self._md5:
            self._md5[md5] = content_path
            return content_path
        return self._md5[md5]

    @staticmethod
    def calc_file_md5(path: str):
        try:
            m = hashlib.md5()
            with open(path, 'rb') as f:
                while True:
                    bits = f.read(4096)
                    if not bits:
                        break
                    m.update(bits)
            return m.hexdigest()
        except Exception as err:
            print(err)
            return ''


def check_dir(config: dict):
    makedirs(config['input_dir'], exist_ok=True)
    makedirs(config['output_dir'], exist_ok=True)


def default_config():
    return {'input_dir': '.app/input_cache',
            'output_dir': '.app/output_cache',
            'md5_name': 'md5.json',
            'test_image_path': 'inputs/cat1.png',
            'test_label': 'plant,grass,cat,stone,other',
            'test_ref_output': 'original_output.npz',
            'test_output_update_name': 'updated_test_output.npz',
            'test_output_name': 'test_output.npz',
            'image_key': 'image',
            'labels_key': 'labels',
            'output_key': 'output',
            'device_name_key': 'device_name',
            'sleep_seconds_for_io': 0.3,
            'result_timeout_in_seconds': 60,
            'image_hw': [360, 480],
            'device': 'hpu',
            'hpu_mode': 1,
            'render_emoji': False}


def load_config(path: str):
    try:
        with open(path, 'r') as f:
            config: Dict = json.load(f)
    except Exception as err:
        print(f"Invalid config: {err}\nLoading default")
        config = default_config()
    return config


def save_config(config: Dict, path: str):
    try:
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as err:
        print(err)


def get_transform(config: Dict):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize(config.get('image_hw', default_config()['image_hw'])),
        ]
    )


def get_utc_time_stamp(fmt: str = '%y-%m-%d-%H-%M-%SZ'):
    return time.strftime(fmt, time.gmtime())


def calc_error(pred: np.ndarray, ref: np.ndarray):
    ae_mat = abs(pred[0] - ref)
    mae = ae_mat.mean()
    se_mat = ae_mat ** 2
    rmse = np.sqrt(se_mat.mean())
    return mae, rmse


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


def get_result_figure(image: Image.Image, labels: List[str], output: np.ndarray,
                      save_path='', title='', alpha=0.5):
    predict = np.argmax(output, 1)
    new_palette = get_new_pallete(len(labels))
    mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=labels)
    seg = mask.convert("RGBA")
    out = Image.blend(image, seg, alpha)
    fig = plt.figure(figsize=(19.2, 3.6))
    axes = fig.subplots(1, 3)
    axes[0].imshow(image)
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
    if save_path != '':
        fig.savefig(save_path)
    return fig
