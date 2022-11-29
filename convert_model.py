# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from os.path import exists, join
import argparse
import numpy as np
from numpy import ndarray
from typing import Optional, Callable

import torch

from modules.lseg_inference import LSegInference
from additional_utils.models import LSeg_MultiEvalModule, LSegMultiEvalAlter

import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms as transforms
from torch.onnx import export as ex_to_onnx


# torch.cuda.device_count()


def get_time_stamp(fmt: str = '%y-%m-%d-%H-%M-%S'):
    return time.strftime(fmt)


def iterate_time(func: Callable, *x_in, n_repeat=10):
    outputs = []
    ts = []
    t0 = time.time()
    for _ in range(n_repeat):
        output = func(*x_in)
        outputs.append(output)
        t1 = time.time()
        ts.append((t1 - t0) * 1000)
        t0 = t1
    return ts, outputs


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
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

        parser.add_argument(
            "--onnx_path", type=str, default="./LANG-SEG.onnx", help="ONNX file path to export"
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args(args=[])
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


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


def calc_loss(pred: ndarray, ref: ndarray, loss_path: Optional[str] = ''):
    ae_mat = abs(pred[0] - ref)
    mae = ae_mat.mean()
    se_mat = ae_mat ** 2
    rse_mat = np.sqrt(se_mat)
    rmse = np.sqrt(se_mat.mean())
    with open(loss_path or 'loss.txt', 'w') as f:
        f.write(f'MAE={mae:g}\n'
                f'AE mat:\n'
                f'{ae_mat}\n'
                f'\n'
                f'RMSE={rmse:g}\n'
                f'RSE mat:\n'
                f'{rse_mat}\n')
    return mae, rmse


def test_onnx(onnx_path: str, image: torch.Tensor, labels: List[str],
              alpha=0.5, save_path='./tmp_onnx.jpg',
              device='cpu', ref: Optional[torch.Tensor] = None,
              loss_path: Optional[str] = ''):
    if device not in ['cpu', 'cuda']:
        device = 'cpu'
    import onnxruntime

    def to_numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    tokens = clip.tokenize(labels)
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device == 'cuda':
        providers = ['CUDAExecutionProvider']
    else:
        raise RuntimeError(f'Invalid `device`: {device}')
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    x = {_in.name: to_numpy(_t) for _in, _t in zip(sess.get_inputs(), (image, tokens))}
    pred = sess.run(None, x)
    if ref is not None:
        mae, rmse = calc_loss(pred[0], to_numpy(ref), loss_path)
        title = f'ONNX inference on {device.upper()}: MAE={mae:.3e}, RMSE={rmse:.3e}'
    else:
        title = ''
    show_result(image, np.argmax(pred[0], 1), labels, alpha, save_path, title)
    return pred


def load_image(image_path='inputs/cat1.jpeg'):
    image = Image.open(image_path)
    image = np.array(image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)
    return image


def load_ref_data(data_path='original_result.npz'):
    try:
        f = np.load(data_path)
        ref = f['output']
    except KeyError:
        f = np.load(data_path)
        ref = f[list(f.keys())[0]]
    except ValueError:
        ref = np.array([])
    return ref


def inference(image_path='inputs/cat1.jpeg', label='plant,grass,cat,stone,other', alpha=0.5,
              to_onnx=True, rewrite_onnx=False, onnx_path='', ref: ndarray = None, out_dir='outputs'):
    """
    Do inference and optionally export to ONNX.

    `ref` (if given) is regarded as the output of Original Torch Model.
    """
    args = Options().parse()

    torch.manual_seed(args.seed)
    args.test_batch_size = 1

    args.scale_inv = False
    args.widehead = True
    args.dataset = 'ade20k'
    args.backbone = 'clip_vitl16_384'
    args.weights = 'checkpoints/demo_e200.ckpt'
    args.ignore_index = 255

    model = LSegInference.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )

    model = model.eval()
    model = model.cpu()
    scales = (
        [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        if args.dataset == "citys"
        else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    evaluator = LSeg_MultiEvalModule(
        model, scales=scales, flip=True
    ).cuda()
    evaluator.eval()
    model.net.init_after_loading()

    image = load_image(image_path)

    print('** Input label value: {} **'.format(label))
    labels = label.split(',')

    if ref is None:
        with torch.no_grad():
            outputs = [evaluator(image.cuda(), clip.tokenize(labels).cuda())]
        title = 'Baseline: Refactored torch model inference on CUDA'
    else:
        outputs = [torch.tensor(ref)]
        title = 'Baseline: Original torch model inference on CUDA'
    predicts = [
        torch.max(output, 1)[1].cpu().numpy()
        for output in outputs
    ]

    predict = predicts[0]

    show_result(image, predict, labels, alpha,
                join(out_dir, f'refactored_inference_{get_time_stamp()}.jpg'), title)
    del evaluator, predict, predicts

    onnx_path: str = onnx_path or args.onnx_path
    if to_onnx and (rewrite_onnx or not exists(onnx_path)):
        model_alter = LSegMultiEvalAlter(model, scales=scales, flip=True, n_class=len(model.net.labels),
                                         sample_input=(image.cuda(), clip.tokenize(labels).cuda())).cuda()
        model_alter.eval()
        del model
        with torch.no_grad():
            scripted_model = torch.jit.script(model_alter)
            del model_alter
            onnx_out = scripted_model(image.cuda(), clip.tokenize(labels).cuda())
            mae, rmse = calc_loss(onnx_out.cpu().numpy(), outputs[0].cpu().numpy(),
                                  join(out_dir, f'compare_script_with_torch_{get_time_stamp()}.txt'))
            title = f'Scripted inference on CUDA: MAE={mae:.3e}, RMSE={rmse:.3e}'
            show_result(image, torch.max(onnx_out, 1)[1].cpu().numpy(), labels, alpha,
                        join(out_dir, f'scripted_inference_{get_time_stamp()}.jpg'), title)
            ex_to_onnx(scripted_model,
                       (image.cuda(), clip.tokenize(labels).cuda()),
                       onnx_path,
                       export_params=True,
                       opset_version=17,
                       do_constant_folding=True,
                       input_names=['image',
                                    'label_tokens'],
                       output_names=['label_map'],
                       dynamic_axes={'image': {2: 'image_h', 3: 'image_w'},
                                     'label_tokens': {0: 'n_tokens'},
                                     'label_map': {1: 'n_tokens', 2: 'image_h', 3: 'image_w'}},
                       verbose=False)
    return outputs


def main():
    image_path = 'inputs/cat1.jpeg'
    label = 'plant,grass,cat,stone,other'
    ref_data_path = './original_output.npz'
    out_dir = './outputs'
    alpha = 0.5
    onnx_path = './LANG-SEG.onnx'
    torch_out = inference(image_path=image_path, label=label, alpha=alpha,
                          to_onnx=True, rewrite_onnx=False, onnx_path=onnx_path,
                          ref=load_ref_data(ref_data_path), out_dir=out_dir)
    print(f'Testing ONNX......')
    device = 'cpu'
    fig_path = join(out_dir, f'./onnx_inference_{device}_{get_time_stamp()}.jpg')
    test_onnx(onnx_path, image=load_image(image_path), labels=label.split(','), alpha=alpha,
              save_path=fig_path, ref=torch_out[0], device=device,
              loss_path=join(out_dir, f'compare_onnx_{device}_with_torch_{get_time_stamp()}.txt'))
    device = 'cuda'
    fig_path = join(out_dir, f'./onnx_inference_{device}_{get_time_stamp()}.jpg')
    test_onnx(onnx_path, image=load_image(image_path), labels=label.split(','), alpha=alpha,
              save_path=fig_path, ref=torch_out[0], device=device,
              loss_path=join(out_dir, f'compare_onnx_{device}_with_torch_{get_time_stamp()}.txt'))
    print(f'Finished testing')


if __name__ == '__main__':
    main()
