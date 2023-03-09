from os import listdir, makedirs, remove
from os.path import basename, dirname, exists, isdir, join
from shutil import move
from time import sleep, time
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from web_app.utils import (Options, get_mask_images_and_object_images,
                           get_preview_figure, get_result_figure, get_utc_time_stamp,
                           load_config, remove_dir_and_files,
                           save_mask_images_and_object_images, singleton, zip_and_save)


@singleton
class LSegFrontend:
    def __init__(self):
        opt = Options().parse()
        self._config = load_config(opt.config_path)
        self._init_session_states()
        self._enable_emoji = self._config.get('render_emoji', False)
        self._data_dir = self._config['input_dir']
        self._out_dir = self._config['output_dir']
        self._create_dirs()
        self._sample_paths = self._get_available_sample_paths()
        self._image: Optional[str, UploadedFile, List[UploadedFile]] = None
        self._label = ''
        self._result: Optional[Tuple[Image.Image, List[str], np.ndarray, str]] = None
        self._supported_devices = self._generate_devices()
        self._target_device = ''
        self._columns: List[DeltaGenerator] = []
        self._test_output_paths = self._generate_device_prefix_paths(self._config['test_output_name'])
        self._test_update_paths = self._generate_device_prefix_paths(self._config['test_update_name'])
        self._refresh_devices()

    def _init_session_states(self):
        states = self.session_states
        if 'disable_interaction' not in states:
            states.disable_interaction = False
        if 'show_result' not in st.session_state:
            states.show_result = True
        if 'last_image_path' not in st.session_state:
            states.last_image_path = ''
        if 'last_time_stamp' not in st.session_state:
            states.last_time_stamp = ''
        if 'last_result_path' not in st.session_state:
            states.last_result_path = ''
        if 'last_download_path' not in st.session_state:
            states.last_download_path = ''

    def _create_dirs(self):
        makedirs(self._data_dir, exist_ok=True)
        makedirs(self._out_dir, exist_ok=True)

    @property
    def session_states(self):
        return st.session_state

    def run(self):
        self._reset_page()
        devices = self._get_available_devices()

        if devices:
            self._run_web_app()
        else:
            self._wait_for_device_preparation()

    def _reset_page(self):
        st.set_page_config(layout="wide")
        st.title(f'{self._emoji("kissing_heart")}Language-guided Semantic Segmentation Web Demo')

    def _run_web_app(self):
        self._select_device()

        self._add_column_layout()
        self._select_image_input_mode()
        self._add_input_widgets()
        self._visualize_input()
        self._add_and_listen_to_process_button()
        self._wait_for_result()
        self._parse_result()
        self._visualize_result()
        self._add_and_listen_to_download_button()

    def _wait_for_device_preparation(self):
        st.markdown(f'{self._emoji("innocent")}:orange[Running initial test]...')
        while True:
            self._check_backend_rerun()

    def _select_device(self):
        devices = self._get_available_devices()
        self._target_device = st.radio(f'{self._emoji("satellite_antenna")}'
                                       f'Select a device:',
                                       devices,
                                       horizontal=True,
                                       disabled=self.session_states.disable_interaction)

    def _add_column_layout(self):
        n = 2
        self._columns.extend(list(st.columns(n)))

    def _select_image_input_mode(self):
        tgt_col = self._columns[1]
        tgt_col.radio(f'{self._emoji("open_mouth")}Choose how to input image:',
                      (f'{self._emoji("point_up_2")}Upload',
                       f'{self._emoji("point_left")}Select from samples',),
                      index=1,
                      key='input_mode',
                      horizontal=True,
                      on_change=self.__hide_result,
                      disabled=self.session_states.disable_interaction)

    def _add_input_widgets(self):
        if self.session_states.input_mode == 'Upload':
            self._add_uploader_for_image_input()
        else:
            self._add_selector_for_image_input()

        self._add_text_input()

    def _add_uploader_for_image_input(self):
        tgt_col = self._columns[0]
        self._image = tgt_col.file_uploader(f'{self._emoji("smirk")}Choose an image...',
                                            on_change=self.__hide_result,
                                            disabled=self.session_states.disable_interaction)

    def _add_selector_for_image_input(self):
        tgt_col = self._columns[0]

        sample_paths = self._get_available_sample_paths()
        tgt_col.image(get_preview_figure(sample_paths))
        option = tgt_col.selectbox(f'{self._emoji("wink")}'
                                   f'Select a sample image by index:',
                                   [str(i) for i in range(len(sample_paths))],
                                   on_change=self.__hide_result,
                                   disabled=self.session_states.disable_interaction)
        self._image = sample_paths[int(option)]

    def _add_text_input(self):
        tgt_col = self._columns[1]
        self._label = tgt_col.text_input(f'{self._emoji("face_with_monocle")}Input labels',
                                         disabled=self.session_states.disable_interaction,
                                         help='Labels split by comma \",\" and '
                                              'each label may consists of multiple space-separated words.\n'
                                              'E.g., \"animal, plant, stone, sky, other\".')

    def _visualize_input(self):
        self._show_image_input()
        self._show_text_input()

    def _add_and_listen_to_process_button(self):
        tgt_col = self._columns[1]

        if tgt_col.button(f'{self._emoji("point_right")}**Start processing**',
                          disabled=self.session_states.disable_interaction):
            self._feed_input()
            st.experimental_rerun()

        if self.session_states.disable_interaction:
            tgt_col.markdown(f'{self._emoji("zany_face")}:green[Started processing...]')

    def _wait_for_result(self):
        states = self.session_states
        if not states.disable_interaction:
            return

        time_stamp = states.last_time_stamp
        if time_stamp == '':
            return
        target_name = self._generate_target_name(time_stamp)
        init_t = time()
        out_dir = self._out_dir
        timeout = self._config['result_timeout_in_seconds']
        progress_bar = st.progress(0.)
        while timeout <= 0 or time() - init_t < timeout:
            progress_bar.progress(min(0.99, (time() - init_t) / timeout))
            for file_name in listdir(out_dir):
                if file_name.startswith(target_name):
                    if exists(states.last_result_path):
                        remove(states.last_result_path)
                    states.last_result_path = join(out_dir, file_name)
                    states.show_result = True
                    progress_bar.progress(1.)
                    sleep(self._config['sleep_seconds_for_io'])
                    return
            if timeout <= 0:
                self._check_backend_rerun()

    def _parse_result(self):
        max_try = 10
        result_path = self.session_states.last_result_path

        for i in range(max_try):
            try:
                with np.load(result_path) as res:
                    image_array: np.ndarray = res[self._config['image_key']]
                    labels: List[str] = res[self._config['labels_key']].tolist()
                    output: np.ndarray = res[self._config['output_key']]
                    device_name: str = res[self._config['device_name_key']]
            except Exception as err:
                print(err)
                if i < max_try - 1:
                    print(f'\nRetry {i + 1}/{max_try - 1}:')
                    sleep(self._config['sleep_seconds_for_io'])
                else:
                    print(f'Failed to parse result {result_path}!')
                    image_hw = self._config['dynamic_image_hw']
                    image_array: np.ndarray = np.zeros((1, *image_hw, 3))
                    labels: List[str] = []
                    output: np.ndarray = np.zeros((1, len(labels), *image_hw))
                    device_name: str = ''
        image = Image.fromarray(
            np.uint8((image_array.squeeze().transpose(1, 2, 0) * 0.5 + 0.5) * 255)
        ).convert('RGBA')

        self._result = (image, labels, output, device_name)

    def _visualize_result(self):
        image, labels, output, device_name = self._result
        fig = get_result_figure(image, labels, output,
                                title=f'{device_name} inference for input'
                                      f' {self._generate_target_name(self.session_states.last_time_stamp)}')
        st.pyplot(fig)

    def _add_and_listen_to_download_button(self):
        tgt_col = self._columns[1]
        states = self.session_states

        self._prepare_download_file()
        download_path = states.last_download_path
        if exists(download_path) and states.show_result:
            with open(download_path, 'rb') as f:
                tgt_col.download_button(f'{self._emoji("heart_eyes")}**Download mask & object images**',
                                        f,
                                        basename(download_path),
                                        mime='application/zip')

    def _check_backend_rerun(self):
        to_rerun = False
        for output_path, update_path, device in zip(self._test_output_paths,
                                                    self._test_update_paths,
                                                    self._supported_devices):
            if exists(update_path):
                move(update_path, output_path)
                to_rerun = True
        if to_rerun:
            self._reset_interaction_state()
            sleep(self._config['sleep_seconds_for_io'])
            st.experimental_rerun()

    def _emoji(self, name: str):
        return f' :{name}: ' if self._enable_emoji else ''

    def _show_image_input(self):
        tgt_col = self._columns[0]

        states = self.session_states
        if states.input_mode == 'Upload':
            if self._image is not None:
                tgt_col.write(f'{self._emoji("sunglasses")}Last uploaded image:')
                tgt_col.image(self._image)
            elif exists(states.last_image_path) and not exists(states.last_result_path):
                tgt_col.write(f'{self._emoji("sunglasses")}Last uploaded image:')
                tgt_col.image(states.last_image_path)
        else:
            if not states.show_result:
                tgt_col.write(f'{self._emoji("sunglasses")}Last selected sample image:')
                tgt_col.image(self._image)

    def _show_text_input(self):
        tgt_col = self._columns[1]
        tgt_col.markdown(f'{self._emoji("thinking_face")}The labels are:\n'
                         f'**:blue[{self._label}]**')

    def _feed_input(self):
        states = self.session_states
        image = self._image or states.last_image_path
        label = self._label
        if image is not None and self._label != '':
            time_stamp = get_utc_time_stamp()
            image = Image.open(image)
            image_path = join(self._data_dir, f'{time_stamp}.png')
            image.save(image_path)
            states.last_image_path = image_path
            input_path = join(self._data_dir, self._generate_target_name(time_stamp))
            with open(input_path, 'w') as f:
                f.write(f'{image_path}\n'
                        f'{label}\n')
            states.last_time_stamp = time_stamp
            self.session_states.disable_interaction = True
        else:
            self._columns[1].markdown(f'{self._emoji("rage")}**:red[Fail to start processing]**')

    def _prepare_download_file(self):
        image, labels, output, device_name = self._result
        states = self.session_states
        if exists(states.last_download_path):
            last_parsed_result_dir = dirname(states.last_download_path)
            remove_dir_and_files(last_parsed_result_dir)

        target_name = self._generate_target_name(states.last_time_stamp)
        parsed_result_dir = join(self._out_dir, target_name)
        makedirs(parsed_result_dir)
        image_paths = [join(parsed_result_dir, 'input_image.png')]
        image.save(image_paths[-1])
        mask_images, object_images = get_mask_images_and_object_images(image, output, len(labels))
        image_paths.extend(save_mask_images_and_object_images(mask_images, object_images,
                                                              labels, parsed_result_dir))
        zip_path = join(parsed_result_dir, f'{target_name}.zip')
        if zip_and_save(zip_path, *image_paths):
            states.last_download_path = zip_path
        else:
            remove_dir_and_files(parsed_result_dir)

    def _reset_interaction_state(self):
        self.session_states.disable_interaction = False

    def __hide_result(self):
        self.session_states.show_result = False

    def _refresh_devices(self):
        for output_path, device in zip(self._test_output_paths,
                                       self._supported_devices):
            if exists(output_path):
                self._supported_devices[device] = True

    def _get_available_sample_paths(self, sample_type=('png',)) -> List[str]:
        if type(sample_type) == str:
            sample_type = (sample_type,)

        sample_dir = self._config.get('sample_dir',
                                      dirname(self._config['test_image_path']))
        sample_paths = []
        if isdir(sample_dir):
            for name in listdir(sample_dir):
                for t in sample_type:
                    if name.endswith(t):
                        sample_paths.append(join(sample_dir, name))
                        break
        return sample_paths

    def _generate_devices(self) -> Dict[str, bool]:
        return {device: False for device in self._config['supported_devices']}

    def _get_available_devices(self):
        return [device for device in self._supported_devices
                if self._supported_devices[device]]

    def _generate_device_prefix_paths(self, base_name: str):
        return [join(self._out_dir, f'{device}_{base_name}')
                for device in self._supported_devices]

    def _generate_target_name(self, time_stamp: str):
        return f'{self._target_device}_{time_stamp}'
