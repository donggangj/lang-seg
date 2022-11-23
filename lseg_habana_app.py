import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from additional_utils.models import LSeg_MultiEvalModule, LSeg_habana_MultiEvalModule
from modules.lseg_inference import LSegInference
# INTEL_CUSTOMIZATION
import habana_frameworks.torch as htorch
if htorch.hpu.is_available():
    # Use hpu as device
    device = torch.device('hpu')
# END of INTEL_CUSTOMIZATION

st.set_page_config(layout="wide")


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


@st.cache(allow_output_mutation=True)
def load_model():
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

            self.parser = parser

        def parse(self):
            opt = self.parser.parse_args(args=[])
            opt.cuda = not opt.no_cuda and torch.cuda.is_available()
            print(opt)
            return opt

    args = Options().parse()

    torch.manual_seed(args.seed)
    args.test_batch_size = 1

    args.scale_inv = False
    args.widehead = True
    args.dataset = 'ade20k'
    args.backbone = 'clip_vitl16_384'
    args.weights = 'checkpoints/demo_e200_fp32.ckpt'
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
    # INTEL_CUSTOMIZATION
    evaluator = LSeg_habana_MultiEvalModule(
        model, scales=scales, flip=True
    ).to(device)
    # END of INTEL_CUSTOMIZATION
    evaluator.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            transforms.Resize([360, 480]),
        ]
    )

    return evaluator, transform


def lseg_demo():
    lseg_model, lseg_transform = load_model()
    # uploaded_file = st.file_uploader("Choose an image...")
    uploaded_file = './inputs/cat1.jpeg'
    input_labels = st.text_input("Input labels", value="plant,grass,cat,stone,other")
    st.write("The labels are", input_labels)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        pimage = lseg_transform(np.array(image)).unsqueeze(0).to(device)

        labels = []
        for label in input_labels.split(","):
            labels.append(label.strip())

        with torch.no_grad():
            output = lseg_model.forward(pimage, labels)
            output_cpu = output.cpu()
            np.savez_compressed('output.npz', output=output_cpu)

            predicts = [
                torch.max(output, 1)[1].cpu().numpy()
                for output in [output]
            ]

        error = calculate_error('output.npz', 'original_output.npz')
        title = 'Habana Gaudi HL-205 inference: MAE={MAE:.3e}, RMSE={RMSE:.3e}'.format(**error)
        show_result(pimage, predicts[0], labels, save_path='result.jpg', title=title)

        image = pimage[0].permute(1, 2, 0)
        image = image * 0.5 + 0.5
        image = Image.fromarray(np.uint8(255 * image.cpu())).convert("RGBA")

        pred = predicts[0]
        new_palette = get_new_pallete(len(labels))
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(seg)
        plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.3, 1), prop={'size': 5})
        plt.axis('off')

        plt.tight_layout()

        st.image([image,seg], width=700, caption=["Input image", "Segmentation"])
        st.pyplot(fig)

def calculate_error(output_path, ref_output_path):
    def mae(ref, pred):
        ref, pred = np.array(ref), np.array(pred)
        return np.mean(np.abs(ref - pred))

    def rmse(ref, pred):
        ref, pred = np.array(ref), np.array(pred)
        return np.sqrt(np.mean((ref-pred)**2))

    error = {}
    ref_output = np.load(ref_output_path)['output']
    output = np.load(output_path)['output']

    error['MAE'] = mae(ref_output, output)
    error['RMSE'] = rmse(ref_output, output)

    return error

def show_result(image, predict, labels: list, save_path: str, title='', alpha=0.5):
    # show results
    image = image.cpu()
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


def main():
    lseg_demo()


if __name__ == '__main__':
    main()
