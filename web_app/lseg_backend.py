from os import remove, listdir, environ
from os.path import join, exists, basename
from time import sleep
from typing import Callable, Sequence

try:
    import habana_frameworks.torch as htorch
except ImportError:
    pass
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from web_app.models import LSeg_habana_MultiEvalModule
from web_app.utils import Options, MD5Table, default_config
from web_app.utils import load_config, check_dir
from lseg.lseg_inference import LSegInference


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
        model, scales=scales
    )
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
        for name in names:
            if '.' in name:
                if self._last_image_t == '' or name.split('.', 1)[0] > self._last_image_t:
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


def get_transform(resize_hw: Sequence[int]):
    if any(value <= 0 for value in resize_hw):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize(resize_hw),
            ]
        )


def get_most_similar_hw(image_hw: Sequence[int], config: dict):
    static_image_size_params = config.get('static_image_size_params',
                                          default_config()['static_image_size_params'])
    hw_ratios = sorted(static_image_size_params['hw_ratios'],
                       key=lambda r: r[0] / r[1])
    hw_ratio = hw_ratios[np.argmin([abs(image_hw[0] / image_hw[1] - r[0] / r[1])
                                    for r in hw_ratios])]
    short_sizes = sorted(static_image_size_params['short_sizes'])
    short_size = short_sizes[np.argmin([abs(min(image_hw) - sz)
                                        for sz in short_sizes])]
    if hw_ratio[0] <= hw_ratio[1]:
        return [short_size, int(short_size / hw_ratio[0] * hw_ratio[1])]
    else:
        return [int(short_size / hw_ratio[1] * hw_ratio[0]), short_size]


def prepare_image(image_path: str, device: torch.device, config: dict):
    image = Image.open(image_path).convert('RGB')
    is_dynamic = not (config.get('device', 'cpu') == 'hpu' and config.get('hpu_mode', 1) == 1)
    if is_dynamic:
        resize_hw = config.get('dynamic_image_hw', default_config()['dynamic_image_hw'])
    else:
        resize_hw = get_most_similar_hw((image.height, image.width), config)
    transform = get_transform(resize_hw)
    return transform(np.array(image)).unsqueeze(0).to(device)


def prepare_input_label(label_str: str, config: dict):
    labels = [label.strip() for label in label_str.split(',')]
    is_dynamic = not (config.get('device', 'cpu') == 'hpu' and config.get('hpu_mode', 1) == 1)
    if is_dynamic:
        if all('other' not in label.lower() for label in labels):
            labels.append('other')
    else:
        labels.extend(['other'] * (config['default_lazy_mode_label_number'] - len(labels)))
    return labels


def prepare_output_label(label_str: str):
    labels = [label.strip() for label in label_str.split(',')]
    return labels


def prepare_torch_device(config: dict):
    if config['device'] == 'hpu' and 'hpu' in torch.__dict__ and torch.hpu.is_available():
        device = torch.device('hpu')
        environ['PT_HPU_LAZY_MODE'] = str(config['hpu_mode'])
        environ['LOG_LEVEL_ALL'] = '0'
        environ['PT_RECIPE_CACHE_PATH'] = './.cache'
        return device
    if config['device'] == 'cuda' and 'cuda' in torch.__dict__ and torch.cuda.is_available():
        device = torch.device('cuda')
        return device
    return torch.device('cpu')


def get_physical_device_name(device: torch.device):
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


def warmup_model_on_device(model: Callable, device: torch.device, config: dict):
    original_image = Image.open(config['test_image_path']).convert('RGB')
    is_dynamic = not (config.get('device', 'cpu') == 'hpu' and config.get('hpu_mode', 1) == 1)
    resize_hw_all = []
    if is_dynamic:
        resize_hw = config.get('dynamic_image_hw', default_config()['dynamic_image_hw'])
        resize_hw_all.append(resize_hw)
    else:
        static_image_size_params = config.get('static_image_size_params',
                                              default_config()['static_image_size_params'])
        hw_ratios = sorted(static_image_size_params['hw_ratios'],
                           key=lambda r: r[0] / r[1])
        short_sizes = sorted(static_image_size_params['short_sizes'])
        resize_hw_all.extend([(int(sz / min(r[:1]) * r[0]), int(sz / min(r[:1]) * r[1]))
                              for r in hw_ratios for sz in short_sizes])
    test_image_all = []
    for resize_hw in resize_hw_all:
        transform = get_transform(resize_hw)
        test_image_all.append(transform(np.array(original_image)).unsqueeze(0).to(device))
    test_labels = prepare_input_label(config['test_label'], config)
    with torch.no_grad():
        test_output_all = [(model(test_image, test_labels)) for test_image in test_image_all]
    return test_image_all, test_labels, test_output_all


def run_backend(opt):
    config = load_config(opt.config_path)
    device = prepare_torch_device(config)
    lseg_model = load_model(opt).to(device)

    # Firstly warmup with test input
    data_dir = config['input_dir']
    out_dir = config['output_dir']
    image_key = config['image_key']
    labels_key = config['labels_key']
    output_key = config['output_key']
    device_name_key = config['device_name_key']
    device_name = get_physical_device_name(device)
    test_image_all, test_labels, test_output_all = warmup_model_on_device(lseg_model, device, config)
    kwargs = {image_key: test_image_all[0].cpu(), labels_key: prepare_output_label(config['test_label']),
              output_key: test_output_all[0].cpu(), device_name_key: device_name}
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
            image = prepare_image(lines[0].strip(), device, config)
            labels = prepare_input_label(lines[1], config)
            with torch.no_grad():
                output = lseg_model(image, labels)
                kwargs = {image_key: image.cpu(), labels_key: prepare_output_label(lines[1]),
                          output_key: output.cpu(), device_name_key: device_name}
                np.savez_compressed(join(out_dir, basename(p)), **kwargs)
                remove(p)


def main():
    opt = BackendOptions().parse()
    check_dir(load_config(opt.config_path))
    run_backend(opt)
