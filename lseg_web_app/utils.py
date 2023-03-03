import argparse
import hashlib
import json
import time
from os import makedirs, listdir, remove, removedirs
from os.path import join, isdir, basename
from typing import Dict, List
from zipfile import ZipFile

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
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


def remove_dir_and_files(dir_path: str):
    for file_name in listdir(dir_path):
        remove(join(dir_path, file_name))
    removedirs(dir_path)


def default_config():
    return {'input_dir': '.app/input_cache',
            'output_dir': '.app/output_cache',
            'md5_name': 'md5.json',
            'sample_dir': 'samples',
            'test_image_path': 'samples/cat1.png',
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
            'dynamic_image_hw': [360, 480],
            "static_image_size_params": {
                "hw_ratios": [
                    [9, 16],
                    [16, 9],
                    [3, 4],
                    [4, 3],
                    [1, 1]
                ],
                "short_sizes": [360, 720, 1080, 1440, 2560]
            },
            'device': 'hpu',
            'hpu_mode': 1,
            'render_emoji': False,
            "default_lazy_mode_label_number": 20}


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


def get_utc_time_stamp(fmt: str = '%y-%m-%d-%H-%M-%SZ'):
    return time.strftime(fmt, time.gmtime())


def calc_error(pred: np.ndarray, ref: np.ndarray):
    ae_mat = abs(pred[0] - ref)
    mae = ae_mat.mean()
    se_mat = ae_mat ** 2
    rmse = np.sqrt(se_mat.mean())
    return mae, rmse


def get_new_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return palette


def get_new_mask_palette(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color palette for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)

    patches = []
    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index < len(labels):
                label = labels[index]
                cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0,
                             new_palette[index * 3 + 2] / 255.0]
                red_patch = mpatches.Patch(color=cur_color, label=label)
                patches.append(red_patch)
    return out_img, patches


def get_result_figure(image: Image.Image, labels: List[str], output: np.ndarray,
                      save_path='', title='', alpha=0.5):
    predict = np.argmax(output, 1)
    new_palette = get_new_palette(len(labels))
    mask, patches = get_new_mask_palette(predict, new_palette, out_label_flag=True, labels=labels)
    seg = mask.convert("RGBA")
    out = Image.blend(image, seg, alpha)
    if len(np.unique(predict)) > len(labels):
        ex_mask_array = (predict >= len(labels)).squeeze()
        ex_mask_array = ex_mask_array.reshape((*ex_mask_array.shape, 1))
        seg = Image.fromarray(seg * np.bitwise_not(ex_mask_array) + image * ex_mask_array)
        out = Image.fromarray(out * np.bitwise_not(ex_mask_array) + image * ex_mask_array)
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


def get_preview_figure(image_paths: List[str]):
    images = [Image.open(p) for p in image_paths]
    max_per_row = 10
    n_image = len(images)
    n_col = min(n_image, max_per_row)
    n_row = (n_image - 1) // n_col + 1
    fig = plt.figure(figsize=(10, 10 * n_row / max_per_row))
    axes = fig.subplots(n_row, n_col)
    if n_row == 1 and n_col == 1:
        axes = [[axes]]
    elif n_row == 1:
        axes = [axes]
    for i in range(n_row):
        for j in range(n_col):
            ax = axes[i][j]
            image_idx = i * n_col + j
            ax.imshow(images[image_idx])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_xlabel(f'{image_idx}')
    return fig


def get_mask_images_and_object_images(image: Image.Image, output: np.ndarray, n_label_in: int):
    n_label_out = output.shape[1]
    predict = np.argmax(output, 1)
    object_mask_array = np.uint8(predict.squeeze())
    mask_image = Image.fromarray(object_mask_array.squeeze().astype('uint8'))
    mask_image.putpalette(get_new_palette(n_label_out))
    mask_image = mask_image.convert('RGBA')
    mask_images = [mask_image]
    if n_label_out > n_label_in:
        ex_mask_array = (predict >= n_label_in).squeeze()
        ex_mask_array = ex_mask_array.reshape((*ex_mask_array.shape, 1))
        mask_images.append(Image.fromarray(mask_image * np.bitwise_not(ex_mask_array) + image * ex_mask_array))
    object_mask_array = object_mask_array.reshape((*object_mask_array.shape, 1))
    object_images = []
    image = image.convert('RGBA')
    for label_id in range(object_mask_array.max() + 1):
        object_images.append(Image.fromarray(image * (object_mask_array == label_id)))
    return mask_images, object_images


def save_mask_images_and_object_images(mask_images: List[Image.Image],
                                       object_images: List[Image.Image],
                                       labels: List[str],
                                       save_dir: str):
    image_paths: List[str] = []
    if not isdir(save_dir) or len(mask_images + object_images) == 0:
        return image_paths
    for mask_id, mask_image in enumerate(mask_images):
        image_paths.append(join(save_dir, f'mask_ver{mask_id}.png'))
        mask_image.save(image_paths[-1])
    for label_id, (label, object_image) in enumerate(zip(labels, object_images)):
        image_paths.append(join(save_dir, f'{label_id}_{label}.png'))
        object_image.save(image_paths[-1])
    return image_paths


def zip_and_save(zip_path: str, *content_paths):
    try:
        with ZipFile(zip_path, 'w') as zip_file:
            for content_path in content_paths:
                zip_file.write(content_path, basename(content_path))
        return True
    except Exception as err:
        print(err)
        return False


def singleton(cls: type):
    _instances = {}

    def get_instance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]
    return get_instance
