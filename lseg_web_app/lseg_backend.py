from os import remove, listdir, environ
from os.path import join, exists, basename
from time import sleep
from typing import Callable

import habana_frameworks.torch as htorch
import numpy as np
import torch
from PIL import Image

from additional_utils.models import LSeg_habana_MultiEvalModule
from lseg_web_app.utils import Options, MD5Table
from lseg_web_app.utils import load_config, check_dir, get_transform
from modules.lseg_inference import LSegInference

if htorch.hpu.is_available():
    # Use hpu as device
    device = torch.device('hpu')
    environ['PT_HPU_LAZY_MODE'] = '1'
    environ['LOG_LEVEL_ALL'] = '0'
    environ['PT_RECIPE_CACHE_PATH'] = './.cache'


class BackendOptions(Options):
    def __init__(self):
        super(BackendOptions, self).__init__()

        parser = self.parser

        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ade20k",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                                training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                                testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default='', help="checkpoint to test"
        )
        # evaluation option
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )
        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )

        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )

        # test option
        parser.add_argument(
            "--data-path", type=str, default='../datasets/', help="path to test image folder"
        )

        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )

        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

    def parse(self):
        opt = self.parser.parse_args(args=[])
        opt.cuda = not opt.no_cuda and torch.cuda.is_available()
        print(opt)
        return opt


def load_model(opt):
    torch.manual_seed(opt.seed)
    opt.test_batch_size = 1

    opt.scale_inv = False
    opt.widehead = True
    opt.dataset = 'ade20k'
    opt.backbone = 'clip_vitl16_384'
    opt.weights = 'checkpoints/demo_e200_fp32.ckpt'
    opt.ignore_index = 255

    model = LSegInference.load_from_checkpoint(
        checkpoint_path=opt.weights,
        data_path=opt.data_path,
        dataset=opt.dataset,
        backbone=opt.backbone,
        aux=opt.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=opt.ignore_index,
        dropout=0.0,
        scale_inv=opt.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=opt.widehead,
        widehead_hr=opt.widehead_hr,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )

    model = model.eval()
    model = model.cpu()
    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if opt.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]
    # INTEL_CUSTOMIZATION
    evaluator = LSeg_habana_MultiEvalModule(
        model, scales=scales, flip=True
    ).to(device)
    # END of INTEL_CUSTOMIZATION
    evaluator.eval()

    return evaluator


class MD5LSeg(MD5Table):
    def __init__(self, md5_table_path: str, data_dir: str):
        super().__init__(md5_table_path)
        self._data_dir = data_dir
        if len(self._md5):
            self._last_image_t = max(basename(p).split('.', 1)[0] for p in self._md5.values())
        else:
            self._last_image_t = ''
        self._last_input_t = self._last_image_t

    def update_md5_table(self, content_path: str):
        md5 = MD5Table.calc_file_md5(content_path)
        old_path = self._md5.get(md5, content_path)
        self._md5[md5] = content_path
        self.save_md5_table()
        return old_path

    def get_latest(self):
        """
        Update image map with duplicated old images removed and return unprocessed inputs
        """
        paths = listdir(self._data_dir)
        md5_name = basename(self._path)
        if md5_name in paths:
            paths.remove(md5_name)
        names = sorted(paths, key=lambda s: ('.' in s, s.split('.', 1)[0]), reverse=True)
        new_image_paths = []
        new_input_paths = []
        for i, name in enumerate(names):
            if '.' in name and (self._last_image_t == '' or name.split('.', 1)[0] > self._last_image_t):
                new_image_paths.append(join(self._data_dir, name))
            else:
                if self._last_input_t == '' or name > self._last_input_t:
                    new_input_paths.append(join(self._data_dir, name))
                else:
                    break
        if len(new_image_paths):
            self._last_image_t = names[0].split('.', 1)[0]
        if len(new_input_paths):
            self._last_input_t = basename(new_input_paths[0])
        new_image_paths.reverse()
        new_input_paths.reverse()
        for image_path in new_image_paths:
            old_path = self.update_md5_table(image_path)
            if old_path != image_path:
                remove(old_path)
        return new_input_paths


def load_image(image_path: str, transform: Callable):
    image = Image.open(image_path).convert('RGB')
    return transform(np.array(image)).unsqueeze(0).to(device)


def load_label(label_str: str):
    return [label.strip() for label in label_str.split(',')]


def get_device_name():
    device_name = 'unknown'
    if device.type == 'cpu':
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'model name' in line:
                device_name = line.split(':')[-1].strip()
                break
    elif device.type == 'cuda':
        device_name = torch.cuda.get_device_name()
    elif device.type == 'hpu':
        device_name = torch.hpu.get_device_name()
    return device_name


def run_backend(opt):
    config = load_config(opt.config_path)
    lseg_model = load_model(opt)

    # Firstly warmup with test input
    transform = get_transform(config)
    image = load_image(config['test_image_path'], transform)
    labels = load_label(config['test_label'])
    data_dir = config['input_dir']
    out_dir = config['output_dir']
    image_key = config['image_key']
    labels_key = config['labels_key']
    output_key = config['output_key']
    device_name_key = config['device_name_key']
    device_name = get_device_name()
    with torch.no_grad():
        output = lseg_model(image, labels)
        kwargs = {image_key: image.cpu(), labels_key: labels,
                  output_key: output.cpu(), device_name_key: device_name}
        test_output_update_path = join(out_dir, config['test_output_update_name'])
        np.savez_compressed(test_output_update_path, **kwargs)

    # Then keep processing until test output is removed
    test_output_path = join(out_dir, config['test_output_name'])
    md5_map = MD5LSeg(join(data_dir, config['md5_name']), data_dir)
    while exists(test_output_path) or exists(test_output_update_path):
        input_paths = md5_map.get_latest()
        sleep(config['sleep_seconds_for_io'])
        for p in input_paths:
            with open(p, 'r') as f:
                lines = f.readlines()
            image = load_image(lines[0].strip(), transform)
            labels = load_label(lines[1])
            with torch.no_grad():
                output = lseg_model(image, labels)
                kwargs = {image_key: image.cpu(), labels_key: labels,
                          output_key: output.cpu(), device_name_key: device_name}
                np.savez_compressed(join(out_dir, basename(p)), **kwargs)
                remove(p)


def main():
    opt = BackendOptions().parse()
    check_dir(load_config(opt.config_path))
    run_backend(opt)
