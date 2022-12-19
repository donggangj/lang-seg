from os import listdir, remove
from os.path import join, exists
from shutil import move
from time import sleep, time

import numpy as np
import streamlit as st
from PIL import Image

from lseg_web_app.utils import Options
from lseg_web_app.utils import load_config, check_dir, get_time_stamp, calc_error, show_result


def init_session_state():
    if 'disable_interaction' not in st.session_state:
        st.session_state['disable_interaction'] = False
    if 'show_test_result' not in st.session_state:
        st.session_state['show_test_result'] = True
    if 'last_image_path' not in st.session_state:
        st.session_state['last_image_path'] = ''
    if 'last_time_stamp' not in st.session_state:
        st.session_state['last_time_stamp'] = ''
    if 'last_result_path' not in st.session_state:
        st.session_state['last_result_path'] = ''


def reset_interaction_state():
    st.session_state['disable_interaction'] = False


def hide_test_result():
    st.session_state['show_test_result'] = False


def get_emoji(emoji_name: str, config: dict):
    return f':{emoji_name}:' if config.get('render_emoji', False) else ''


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
    sleep(config['sleep_seconds_for_io'])
    with np.load(test_output_path) as res:
        mae, rmse = calc_error(res[config['output_key']],
                               np.load(config['test_ref_output'])[config['output_key']])
        device_name = res[config['device_name_key']]
    title = f'Test {device_name} inference: MAE={mae:g}, RMSE={rmse:g}'
    fig = show_result(test_output_path, config, title=title)
    st.pyplot(fig)


def feed_inputs(image, label: str, data_dir: str):
    if image is not None and label != '':
        time_stamp = get_time_stamp()
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
                st.session_state['show_test_result'] = False
                progress_bar.progress(1.)
                sleep(config['sleep_seconds_for_io'])
                return True
        if timeout <= 0:
            check_backend_rerun(config)
    return False


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
    st.title(f'Language-guided Semantic Segmentation Web Demo{get_emoji("kissing_heart", config)}')
    check_backend_rerun(config)

    data_dir = config['input_dir']
    out_dir = config['output_dir']
    test_output_path = join(out_dir, config['test_output_name'])
    if exists(test_output_path):
        if st.session_state['show_test_result']:
            st.markdown(f':green[Initial test result]{get_emoji("hugging_face", config)}:')
            show_test_result(config)
        col1, col2 = st.columns(2)
        uploaded = col1.file_uploader(f'Choose an image...{get_emoji("smirk", config)}',
                                      on_change=hide_test_result,
                                      disabled=st.session_state['disable_interaction'])
        if uploaded is not None:
            col1.write('Last uploaded image:')
            col1.image(uploaded)
        elif exists(st.session_state['last_image_path']) and not exists(st.session_state['last_result_path']):
            col1.write('Last uploaded image:')
            col1.image(st.session_state['last_image_path'])
        label = col2.text_input(f'Input labels{get_emoji("face_with_monocle", config)}',
                                disabled=st.session_state['disable_interaction'])
        col2.markdown(f'{get_emoji("thinking_face", config)} The labels are:\n**:blue[{label}]**')
        if col2.button(f'{get_emoji("point_right", config)}**Start processing**',
                       disabled=st.session_state['disable_interaction']):
            if feed_inputs(uploaded or st.session_state['last_image_path'],
                           label, data_dir):
                st.session_state['disable_interaction'] = True
            else:
                col2.markdown(f'**:red[Fail to start processing]**{get_emoji("rage", config)}')
            st.experimental_rerun()
        if st.session_state['disable_interaction'] is True:
            col2.markdown(f':green[Started processing...]{get_emoji("zany_face", config)}')
            update_result(config)
            st.session_state['disable_interaction'] = False
            st.experimental_rerun()
        if st.session_state['last_result_path']:
            fig = show_result(st.session_state['last_result_path'], config)
            st.pyplot(fig)
    else:
        st.write('Running initial test...')
        while True:
            check_backend_rerun(config)


def main():
    opt = Options().parse()
    check_dir(load_config(opt.config_path))
    init_session_state()
    run_frontend(opt)
