from os import listdir, remove
from os.path import join, exists
from shutil import move
from time import sleep

import numpy as np
import streamlit as st
from PIL import Image

from lseg_web_app.utils import Options
from lseg_web_app.utils import load_config, check_dir, get_time_stamp, calc_error, show_result


def check_update(config: dict):
    out_dir = config['output_dir']
    test_output_update_path = join(out_dir, config['test_output_update_name'])
    if exists(test_output_update_path):
        test_output_path = join(out_dir, config['test_output_name'])
        move(test_output_update_path, test_output_path)
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


def run_frontend(opt):
    st.set_page_config(layout="wide")
    config = load_config(opt.config_path)
    check_update(config)

    data_dir = config['input_dir']
    out_dir = config['output_dir']
    test_output_path = join(out_dir, config['test_output_name'])
    st.write('Testing...')
    if exists(test_output_path):
        st.write('Test result:')
        show_test_result(config)
        uploaded = st.file_uploader("Choose an image...")
        if uploaded is not None:
            st.write('Uploaded image:')
            st.image(uploaded)
        label = st.text_input("Input labels")
        if uploaded is not None and label != '':
            name = get_time_stamp()
            image = Image.open(uploaded)
            image_path = join(data_dir, f'{name}.jpg')
            image.save(image_path)
            input_path = join(data_dir, name)
            with open(input_path, 'w') as f:
                f.write(f'{image_path}\n'
                        f'{label}\n')
            res = []
            while exists(input_path):
                res = listdir(out_dir)
                res.remove(config['test_output_name'])
                if len(res):
                    sleep(config['sleep_seconds_for_io'])
            for res_name in res:
                res_path = join(out_dir, res_name)
                fig = show_result(res_path, config)
                st.pyplot(fig)
                remove(res_path)
    else:
        while True:
            check_update(config)


def main():
    opt = Options().parse()
    check_dir(load_config(opt.config_path))
    run_frontend(opt)
