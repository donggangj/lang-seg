from os import listdir, remove, makedirs, removedirs
from os.path import join, exists, basename, dirname, isdir
from shutil import move
from time import sleep, time
from typing import List, Union
from zipfile import ZipFile

import numpy as np
import streamlit as st
from PIL import Image

from lseg_web_app.utils import (Options, load_config, check_dir, get_utc_time_stamp, calc_error,
                                get_result_figure, get_preview_figure)


def init_session_state():
    if 'disable_interaction' not in st.session_state:
        st.session_state['disable_interaction'] = False
    if 'show_result' not in st.session_state:
        st.session_state['show_result'] = True
    if 'last_image_path' not in st.session_state:
        st.session_state['last_image_path'] = ''
    if 'last_time_stamp' not in st.session_state:
        st.session_state['last_time_stamp'] = ''
    if 'last_result_path' not in st.session_state:
        st.session_state['last_result_path'] = ''
    if 'last_download_path' not in st.session_state:
        st.session_state['last_download_path'] = ''


def reset_interaction_state():
    st.session_state['disable_interaction'] = False


def hide_result():
    st.session_state['show_result'] = False


def get_emoji(emoji_name: str, config: dict):
    return f' :{emoji_name}: ' if config.get('render_emoji', False) else ''


def check_backend_rerun(config: dict):
    out_dir = config['output_dir']
    test_output_update_path = join(out_dir, config['test_output_update_name'])
    if exists(test_output_update_path):
        test_output_path = join(out_dir, config['test_output_name'])
        move(test_output_update_path, test_output_path)
        sleep(config['sleep_seconds_for_io'])
        reset_interaction_state()
        st.experimental_rerun()


def show_test_result(config: dict):
    out_dir = config['output_dir']
    test_output_path = join(out_dir, config['test_output_name'])
    image, labels, output, device_name = parse_result(test_output_path, config)
    assert output.size > 0
    mae, rmse = calc_error(output,
                           np.load(config['test_ref_output'])[config['output_key']])
    title = f'Test {device_name} inference: MAE={mae:g}, RMSE={rmse:g}'
    fig = get_result_figure(image, labels, output, title=title)
    st.pyplot(fig)


def get_available_sample_paths(config: dict, sample_type=('png',)) -> List[str]:
    sample_dir = config.get('sample_dir', dirname(config['test_image_path']))
    sample_paths = []
    if not isdir(sample_dir):
        return sample_paths
    if type(sample_type) == str:
        sample_type = (sample_type,)
    for name in listdir(sample_dir):
        for t in sample_type:
            if name.endswith(t):
                sample_paths.append(join(sample_dir, name))
                break
    return sample_paths


def feed_inputs(image: Union[Image.Image, str], label: str, data_dir: str):
    if image is not None and label != '':
        time_stamp = get_utc_time_stamp()
        image = Image.open(image)
        image_path = join(data_dir, f'{time_stamp}.png')
        image.save(image_path)
        st.session_state['last_image_path'] = image_path
        input_path = join(data_dir, time_stamp)
        with open(input_path, 'w') as f:
            f.write(f'{image_path}\n'
                    f'{label}\n')
        st.session_state['last_time_stamp'] = time_stamp
        return True
    return False


def update_result(config: dict):
    target_name = st.session_state['last_time_stamp']
    if target_name == '':
        return False
    init_t = time()
    out_dir = config['output_dir']
    timeout = config['result_timeout_in_seconds']
    progress_bar = st.progress(0.)
    while timeout <= 0 or time() - init_t < timeout:
        progress_bar.progress(min(0.99, (time() - init_t) / timeout))
        for file_name in listdir(out_dir):
            if target_name == file_name.split('.', 1)[0]:
                if exists(st.session_state['last_result_path']):
                    remove(st.session_state['last_result_path'])
                st.session_state['last_result_path'] = join(out_dir, file_name)
                st.session_state['show_result'] = True
                progress_bar.progress(1.)
                sleep(config['sleep_seconds_for_io'])
                return True
        if timeout <= 0:
            check_backend_rerun(config)
    return False


def parse_result(res_path: str, config: dict):
    max_try = 3
    for i in range(max_try):
        try:
            with np.load(res_path) as res:
                image_array: np.ndarray = res[config['image_key']]
                labels: List[str] = res[config['labels_key']].tolist()
                output: np.ndarray = res[config['output_key']]
                device_name: str = res[config['device_name_key']]
        except Exception as err:
            print(err)
            if i < max_try - 1:
                print(f'\nRetry {i + 1}/{max_try - 1}:')
                sleep(config['sleep_seconds_for_io'])
            else:
                print(f'Failed to parse result {res_path}!')
                image_array: np.ndarray = np.zeros((1, *config['image_hw'], 3))
                labels: List[str] = []
                output: np.ndarray = np.asarray([])
                device_name: str = ''
    image = Image.fromarray(
        np.uint8((image_array.squeeze().transpose(1, 2, 0) * 0.5 + 0.5) * 255)
    ).convert('RGBA')
    return image, labels, output, device_name


def get_mask_and_object_images(image: Image.Image, output: np.ndarray):
    mask_array = np.uint8(np.argmax(output, 1).squeeze())
    mask_image = Image.fromarray(mask_array)
    mask_and_object_images = [mask_image]
    mask_array = mask_array.reshape((*mask_array.shape, 1))
    image = image.convert('RGBA')
    for label_id in range(mask_array.max() + 1):
        mask_and_object_images.append(Image.fromarray(image * (mask_array == label_id)))
    return mask_and_object_images


def save_mask_and_object_images(mask_and_object_images: List[Image.Image],
                                labels: List[str],
                                save_dir: str):
    image_paths: List[str] = []
    if not isdir(save_dir) or len(mask_and_object_images) == 0:
        return image_paths
    mask_image = mask_and_object_images[0]
    object_images = mask_and_object_images[1:]
    image_paths.append(join(save_dir, 'mask.png'))
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


def prepare_download_file(image: Image.Image, labels: List[str], output: np.ndarray, config: dict):
    def remove_dir_and_files(dir_path: str):
        for file_name in listdir(dir_path):
            remove(join(dir_path, file_name))
        removedirs(dir_path)

    if exists(st.session_state['last_download_path']):
        last_parsed_result_dir = dirname(st.session_state['last_download_path'])
        remove_dir_and_files(last_parsed_result_dir)

    parsed_result_dir = join(config['output_dir'], st.session_state['last_time_stamp'])
    makedirs(parsed_result_dir)
    image_paths = [join(parsed_result_dir, 'input_image.png')]
    image.save(image_paths[-1])
    mask_and_object_images = get_mask_and_object_images(image, output)
    image_paths.extend(save_mask_and_object_images(mask_and_object_images, labels, parsed_result_dir))
    zip_path = join(parsed_result_dir, f'{st.session_state["last_time_stamp"]}.zip')
    if zip_and_save(zip_path, *image_paths):
        st.session_state['last_download_path'] = zip_path
    else:
        remove_dir_and_files(parsed_result_dir)


def fetch_results(config: dict):
    init_t = time()
    out_dir = config['output_dir']
    timeout = config['result_timeout_in_seconds']
    while timeout <= 0 or time() - init_t < timeout:
        res = listdir(out_dir)
        res.remove(config['test_output_name'])
        if config['test_output_update_name'] in res:
            res.remove(config['test_output_update_name'])
        if len(res):
            sleep(config['sleep_seconds_for_io'])
            return res
        if timeout <= 0:
            check_backend_rerun(config)
    return []


def run_frontend(opt):
    st.set_page_config(layout="wide")
    config = load_config(opt.config_path)
    st.title(f'{get_emoji("kissing_heart", config)}Language-guided Semantic Segmentation Web Demo')
    check_backend_rerun(config)

    data_dir = config['input_dir']
    out_dir = config['output_dir']
    test_output_path = join(out_dir, config['test_output_name'])
    if exists(test_output_path):
        if st.session_state['show_result'] and not exists(st.session_state['last_result_path']):
            st.markdown(f'{get_emoji("hugging_face", config)}:green[Initial test result]:')
            show_test_result(config)
        col1, col2 = st.columns(2)
        col2.radio(f'{get_emoji("open_mouth", config)}Choose how to input image:',
                   ('Upload', 'Select from samples'),
                   key='input_selection',
                   horizontal=True,
                   on_change=hide_result,
                   disabled=st.session_state['disable_interaction'])
        if st.session_state['input_selection'] == 'Upload':
            uploaded = col1.file_uploader(f'{get_emoji("smirk", config)}Choose an image...',
                                          on_change=hide_result,
                                          disabled=st.session_state['disable_interaction'])
            if uploaded is not None:
                col1.write(f'{get_emoji("sunglasses", config)}Last uploaded image:')
                col1.image(uploaded)
            elif exists(st.session_state['last_image_path']) and not exists(st.session_state['last_result_path']):
                col1.write(f'{get_emoji("sunglasses", config)}Last uploaded image:')
                col1.image(st.session_state['last_image_path'])
        else:
            sample_paths = get_available_sample_paths(config)
            col1.pyplot(get_preview_figure(sample_paths))
            option = col1.selectbox(f'{get_emoji("wink", config)}'
                                    f'Select a sample image by index:',
                                    [str(i) for i in range(len(sample_paths))],
                                    on_change=hide_result,
                                    disabled=st.session_state['disable_interaction'])
            uploaded = sample_paths[int(option)]
            if not st.session_state['show_result']:
                col1.write(f'{get_emoji("sunglasses", config)}Last selected sample image:')
                col1.image(uploaded)
        label = col2.text_input(f'{get_emoji("face_with_monocle", config)}Input labels',
                                disabled=st.session_state['disable_interaction'],
                                help='Labels split by comma \",\" and '
                                     'each label may consists of multiple space-separated words.\n'
                                     'E.g., \"animal, plant, stone, sky, other\".')
        col2.markdown(f'{get_emoji("thinking_face", config)}The labels are:\n**:blue[{label}]**')
        if col2.button(f'{get_emoji("point_right", config)}**Start processing**',
                       disabled=st.session_state['disable_interaction']):
            if feed_inputs(uploaded or st.session_state['last_image_path'],
                           label, data_dir):
                st.session_state['disable_interaction'] = True
            else:
                col2.markdown(f'{get_emoji("rage", config)}**:red[Fail to start processing]**')
            st.experimental_rerun()
        if st.session_state['disable_interaction'] is True:
            col2.markdown(f'{get_emoji("zany_face", config)}:green[Started processing...]')
            update_result(config)
            st.session_state['disable_interaction'] = False
            st.experimental_rerun()
        if st.session_state['show_result'] and exists(st.session_state['last_result_path']):
            image, labels, output, device_name = parse_result(st.session_state['last_result_path'],
                                                              config)
            if output.size > 0:
                prepare_download_file(image, labels, output, config)
                fig = get_result_figure(image, labels, output,
                                        title=f'{device_name} inference for input'
                                              f' {basename(st.session_state["last_result_path"]).rsplit(".", 1)[0]}')
                st.pyplot(fig)
        if exists(st.session_state["last_download_path"]):
            with open(st.session_state["last_download_path"], 'rb') as f:
                col2.download_button(f'{get_emoji("heart_eyes", config)}**Download mask & object images**',
                                     f,
                                     basename(st.session_state["last_download_path"]),
                                     mime='application/zip')
    else:
        st.markdown(f'{get_emoji("innocent", config)}:orange[Running initial test]...')
        while True:
            check_backend_rerun(config)


def main():
    opt = Options().parse()
    check_dir(load_config(opt.config_path))
    init_session_state()
    run_frontend(opt)
