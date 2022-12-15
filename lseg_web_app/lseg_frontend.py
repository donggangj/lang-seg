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
    if 'has_result' not in st.session_state:
        st.session_state['has_result'] = False
    if 'last_image_path' not in st.session_state:
        st.session_state['last_image_path'] = ''


def reset_session_state():
    st.session_state['disable_interaction'] = False
    st.session_state['has_result'] = False


def check_update(config: dict):
    out_dir = config['output_dir']
    test_output_update_path = join(out_dir, config['test_output_update_name'])
    if exists(test_output_update_path):
        test_output_path = join(out_dir, config['test_output_name'])
        move(test_output_update_path, test_output_path)
        sleep(config['sleep_seconds_for_io'])
        reset_session_state()
        st.experimental_rerun()


def show_test_result(config: dict):
    out_dir = config['output_dir']
    test_output_path = join(out_dir, config['test_output_name'])
    sleep(config['sleep_seconds_for_io'])
    res = np.load(test_output_path)
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
        return True
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
            check_update(config)
    return []


def run_frontend(opt):
    st.set_page_config(layout="wide")
    st.title('Language-guided Semantic Segmentation Web Demo')
    config = load_config(opt.config_path)
    check_update(config)

    data_dir = config['input_dir']
    out_dir = config['output_dir']
    test_output_path = join(out_dir, config['test_output_name'])
    if exists(test_output_path):
        st.write('Initial test result:')
        show_test_result(config)
        col1, col2 = st.columns(2)
        uploaded = col1.file_uploader("Choose an image...",
                                      disabled=st.session_state['disable_interaction'])
        if uploaded is not None:
            col1.write('Last uploaded image:')
            col1.image(uploaded)
        elif exists(st.session_state['last_image_path']) and not st.session_state['has_result']:
            col1.write('Last uploaded image:')
            col1.image(st.session_state['last_image_path'])
        label = col2.text_input("Input labels",
                                disabled=st.session_state['disable_interaction'])
        col2.write(f'The labels are:\n{label}')
        if col2.button('Start processing',
                       disabled=st.session_state['disable_interaction']):
            if feed_inputs(uploaded or st.session_state['last_image_path'],
                           label, data_dir):
                st.session_state['disable_interaction'] = True
            else:
                col2.write('Fail to start processing')
            st.experimental_rerun()
        if st.session_state['disable_interaction'] is True:
            st.session_state['has_result'] = True
            col2.write('Started processing...')
            fetch_results(config)
            st.session_state['disable_interaction'] = False
            st.experimental_rerun()
        if st.session_state['has_result']:
            res = fetch_results(config)
            st.session_state['has_result'] = False
        else:
            res = []
        for res_name in res:
            res_path = join(out_dir, res_name)
            fig = show_result(res_path, config)
            st.pyplot(fig)
            remove(res_path)
    else:
        st.write('Running initial testing...')


def main():
    opt = Options().parse()
    check_dir(load_config(opt.config_path))
    init_session_state()
    run_frontend(opt)
