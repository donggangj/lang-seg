from os.path import join

import clip
import numpy as np
import torch
from openvino.runtime import Core

from utils.data import prepare_image, prepare_label, load_ref_data
from utils.info import get_time_stamp, get_physical_device_name, show_result
from utils.performance import iterate_time, calc_loss


def test_onnx(onnx_path: str, image_path: str, label: str, ref_path='',
              alpha=0.5, device='cpu', save_path='./tmp_onnx.jpg'):
    if device not in ['cpu', 'cuda']:
        device = 'cpu'
    import onnxruntime

    labels = prepare_label(label)
    tokens = clip.tokenize(labels)
    if device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif device == 'cuda':
        providers = ['CUDAExecutionProvider']
    else:
        raise RuntimeError(f'Invalid `device`: {device}')
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    image = prepare_image(image_path)
    x = {_in.name: _t for _in, _t in zip(sess.get_inputs(), (image, tokens))}
    pred = sess.run(None, x)
    if ref_path:
        mae, rmse = calc_loss(pred[0], load_ref_data(ref_path))
        title = f'ONNX inference on {device.upper()}: MAE={mae:.3e}, RMSE={rmse:.3e}'
    else:
        title = f'ONNX inference on {device.upper()}'
    show_result(image, np.argmax(pred[0], 1), labels, alpha, save_path, title)
    return pred


def test_ov(ir_path: str, image_path: str, label: str, ref_path='',
            alpha=0.5, device='cpu',
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
    image = prepare_image(image_path)
    labels = prepare_label(label)
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
    if ref_path:
        mae, rmse = calc_loss(out, load_ref_data(ref_path))
        title = f'OpenVINO inference on {device_name}: MAE={mae:.3e}, RMSE={rmse:.3e}'
    else:
        title = f'OpenVINO inference on {device_name}'
    fig_path = join(out_dir, f'openvino_inference_{get_time_stamp()}.jpg')
    show_result(image, np.argmax(out, 1), labels, alpha, fig_path, title)


def main(test_mode=0):
    samples = [
        {'image_path': './samples/cat1.png',
         'label': 'plant,grass,cat,stone,other',
         'ref_path': './samples/cat1.npz'},
        {'image_path': './samples/cat1.png',
         'label': 'plant,cat,stone,other',
         'ref_path': ''},
        {'image_path': './samples/ADE_val_00000001.png',
         'label': 'house,grass,sky,other',
         'ref_path': ''},
        {'image_path': './samples/ADE_val_00000001.png',
         'label': 'house,sky,other',
         'ref_path': ''},
        {'image_path': './samples/ADE_val_00000001.png',
         'label': 'house,sky,grass,wall,other',
         'ref_path': ''}
    ]
    out_dir = './outputs'
    alpha = 0.5
    onnx_path = './LANG-SEG-opset16.onnx'
    ir_path = './ov_py39/LANG-SEG_opset16.xml'
    if test_mode != -1:
        samples = samples[:1]
        n_repeat = 10
    else:
        n_repeat = -1

    if test_mode == 0 or test_mode == -1:
        device = 'cpu'
        print(f'Testing ONNX on {device}......')
        for sample in samples:
            fig_path = join(out_dir, f'./onnx_inference_{device}_{get_time_stamp()}.jpg')
            test_onnx(onnx_path, **sample, alpha=alpha,
                      save_path=fig_path, device=device)
        print(f'Finished testing')
    if test_mode == 1 or test_mode == -1:
        device = 'cuda'
        print(f'Testing ONNX on {device}......')
        for sample in samples:
            fig_path = join(out_dir, f'./onnx_inference_{device}_{get_time_stamp()}.jpg')
            test_onnx(onnx_path, **sample, alpha=alpha,
                      save_path=fig_path, device=device)
        print(f'Finished testing')
    if test_mode == 2 or test_mode == -1:
        device = 'CPU'
        print(f'Testing OpenVINO on {device}......')
        for sample in samples:
            test_ov(ir_path, **sample, alpha=alpha, device=device, n_repeat=n_repeat)
        print(f'Finished testing')
    if test_mode == 3 or test_mode == -1:
        device = 'GPU.0'
        print(f'Testing OpenVINO on {device}......')
        for sample in samples:
            test_ov(ir_path, **sample, alpha=alpha, device=device, n_repeat=n_repeat)
        print(f'Finished testing')
    if test_mode == 4 or test_mode == -1:
        device = 'GPU.1'
        print(f'Testing OpenVINO on {device}......')
        for sample in samples:
            test_ov(ir_path, **sample, alpha=alpha, device=device, n_repeat=n_repeat)
        print(f'Finished testing')


if __name__ == '__main__':
    main(4)
