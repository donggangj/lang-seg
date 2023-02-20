import time
from os.path import join
from typing import Callable

import clip
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from numpy import ndarray
from openvino.runtime import Core


def get_time_stamp(fmt: str = '%y-%m-%d-%H-%M-%SZ'):
    return time.strftime(fmt, time.gmtime())


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
        available_gpus = [_device for _device in core.available_devices if 'gpu' in _device]
        if device in available_gpus:
            device_name = core.get_property(device_name, "FULL_DEVICE_NAME")
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


def calc_loss(pred: ndarray, ref: ndarray):
    ae_mat = abs(pred[0] - ref)
    mae = ae_mat.mean()
    se_mat = ae_mat ** 2
    rmse = np.sqrt(se_mat.mean())
    return mae, rmse


def test_onnx(onnx_path: str, image_path: str, label: str,
              alpha=0.5, save_path='./tmp_onnx.jpg',
              device='cpu', ref: ndarray = None):
    if device not in ['cpu', 'cuda']:
        device = 'cpu'
    import onnxruntime

    labels = label.split(',')
    tokens = clip.tokenize(labels)
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device == 'cuda':
        providers = ['CUDAExecutionProvider']
    else:
        raise RuntimeError(f'Invalid `device`: {device}')
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    image = load_image(image_path)
    x = {_in.name: _t for _in, _t in zip(sess.get_inputs(), (image, tokens))}
    pred = sess.run(None, x)
    if ref is not None:
        mae, rmse = calc_loss(pred[0], ref)
        title = f'ONNX inference on {device.upper()}: MAE={mae:.3e}, RMSE={rmse:.3e}'
    else:
        title = f'ONNX inference on {device.upper()}'
    show_result(image, np.argmax(pred[0], 1), labels, alpha, save_path, title)
    return pred


def test_ov(ir_path: str, image_path: str, label: str, alpha=0.5,
            device='cpu', ref: ndarray = None,
            out_dir='outputs', log_dir='logs', n_repeat: int = -1):
    def to_numpy(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    if device.lower().startswith('gpu'):
        device = device.upper()
    else:
        device = 'CPU'
    device_name = get_physical_device_name(device.lower())
    core = Core()
    ir_model = core.read_model(ir_path)
    model = core.compile_model(ir_model, device)
    image = load_image(image_path)
    labels = label.split(',')
    tokens = clip.tokenize(labels)
    x_in = ((to_numpy(image), to_numpy(tokens)),)
    if n_repeat <= 0:
        out = model.infer_new_request(*x_in)
        out = next(iter(out.values()))
    else:
        log_path = join(log_dir, f'openvino_inference_time_{get_time_stamp()}.log')
        ts, outputs = iterate_time(model.infer_new_request, *x_in, n_repeat=n_repeat + 1)
        out = next(iter(outputs[0].values()))
        with open(log_path, 'w') as f:
            f.write(f'image_path: {image_path}\n'
                    f'label: {label}\n'
                    f'device: {device_name}\n'
                    f'n_repeat: {n_repeat}\n'
                    f'mean inference time (ms): {sum(ts[-n_repeat:]) / n_repeat:.3e}\n'
                    f'starting time (ms): {ts[:-n_repeat]}\n'
                    f'inference time (ms):\n{ts[-n_repeat:]}\n')
    if ref is not None:
        mae, rmse = calc_loss(out, ref)
        title = f'OpenVINO inference on {device_name}: MAE={mae:.3e}, RMSE={rmse:.3e}'
    else:
        title = f'OpenVINO inference on {device_name}'
    fig_path = join(out_dir, f'openvino_inference_{get_time_stamp()}.jpg')
    show_result(image, np.argmax(out, 1), labels, alpha, fig_path, title)


def load_image(image_path='samples/cat1.jpeg'):
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


def main(test_mode=0):
    samples = [
        {'image_path': './samples/cat1.jpeg',
         'label': 'plant,grass,cat,stone,other'},
        {'image_path': './samples/cat1.jpeg',
         'label': 'plant,cat,stone,other'},
        {'image_path': './samples/ADE_val_00000001.jpg',
         'label': 'house,grass,sky,other'},
        {'image_path': './samples/ADE_val_00000001.jpg',
         'label': 'house,sky,other'},
        {'image_path': './samples/ADE_val_00000001.jpg',
         'label': 'house,sky,grass,wall,other'}
    ]
    out_dir = './outputs'
    alpha = 0.5
    onnx_path = './LANG-SEG-opset16.onnx'
    ir_path = './ov_py39/LANG-SEG_opset16.xml'
    ref_data_path = './original_output.npz'
    if test_mode != -1:
        samples = samples[:1]
        ref = None
        n_repeat = 10
    else:
        ref = load_ref_data(ref_data_path)
        n_repeat = -1

    if test_mode == 0 or test_mode == -1:
        device = 'cpu'
        print(f'Testing ONNX on {device}......')
        for sample in samples:
            fig_path = join(out_dir, f'./onnx_inference_{device}_{get_time_stamp()}.jpg')
            test_onnx(onnx_path, **sample, alpha=alpha,
                      save_path=fig_path, ref=ref, device=device)
        print(f'Finished testing')
    if test_mode == 1 or test_mode == -1:
        device = 'cuda'
        print(f'Testing ONNX on {device}......')
        for sample in samples:
            fig_path = join(out_dir, f'./onnx_inference_{device}_{get_time_stamp()}.jpg')
            test_onnx(onnx_path, **sample, alpha=alpha,
                      save_path=fig_path, ref=ref, device=device)
        print(f'Finished testing')
    if test_mode == 2 or test_mode == -1:
        device = 'CPU'
        print(f'Testing OpenVINO on {device}......')
        for sample in samples:
            test_ov(ir_path, **sample, alpha=alpha, device=device, ref=ref, n_repeat=n_repeat)
        print(f'Finished testing')
    if test_mode == 3 or test_mode == -1:
        device = 'GPU.0'
        print(f'Testing OpenVINO on {device}......')
        for sample in samples:
            test_ov(ir_path, **sample, alpha=alpha, device=device, ref=ref, n_repeat=n_repeat)
        print(f'Finished testing')
    if test_mode == 4 or test_mode == -1:
        device = 'GPU.1'
        print(f'Testing OpenVINO on {device}......')
        for sample in samples:
            test_ov(ir_path, **sample, alpha=alpha, device=device, ref=ref, n_repeat=n_repeat)
        print(f'Finished testing')


if __name__ == '__main__':
    main(4)
