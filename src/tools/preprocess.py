import os
from tools.utils import read_file_contents_list, mkdir_p, convert_3d_2_flat
from tools.paral import AbstractParallelRoutine
from tools.data_io import DataFolder, ScanWrapper
import numpy as np
import time


class PreprocessDownSample(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, ref_img, order=3):
        super().__init__(config, in_folder)
        self._c3d = config['c3d_exe']
        self._file_list = self._get_file_list(config['data_file_list'])
        self._spacing_config = config['spacing']
        self._in_folder = in_folder
        self._out_folder = out_folder
        mkdir_p(out_folder)
        self._order = order
        self._reg_resample = config['niftyreg_resample']
        self._reg_resmaple_ref = config['reg_resample_ref_img']

    def _run_chunk(self, chunk_list):
        for id_file in chunk_list:
            self._in_data_folder.print_idx(id_file)
            in_img_name = self._in_data_folder.get_file_name(id_file)
            in_img_path = self._in_data_folder.get_file_path(id_file)
            out_img_path = os.path.join(self._out_folder, in_img_name)
            c3d_cmd_str = self._get_c3d_cmd_str(in_img_path, out_img_path)
            print(c3d_cmd_str, flush=True)
            os.system(c3d_cmd_str)

    def run(self):
        for idx in range(self.num_files()):
            file_name = self._file_list[idx]
            print(f'Preprocess scan {file_name} ({idx}/{self.num_files()})', flush=True)
            self._run_single_file(file_name)
            print('Done', flush=True)

    def num_files(self):
        return len(self._file_list)

    def _run_single_file(self, file_name):
        in_file_path = os.path.join(self._in_folder, file_name)
        out_file_path = os.path.join(self._out_folder, file_name)

        # cmd_str = self._get_c3d_cmd_str(in_file_path, out_file_path)
        cmd_str = self._get_niftyreg_cmd_str(in_file_path, out_file_path)
        print(cmd_str, flush=True)
        os.system(cmd_str)

    def _get_spacing_cmd(self):
        spacing = {}
        spacing[0] = self._spacing_config['X']
        spacing[1] = self._spacing_config['Y']
        spacing[2] = self._spacing_config['Z']
        spacing_str = f'-resample-mm {spacing[0]}x{spacing[1]}x{spacing[2]}mm'

        return spacing_str

    def _get_c3d_cmd_str(self, in_file_path, out_file_path):
        spacing_str = self._get_spacing_cmd()
        cmd_str_resample = f'{self._c3d} {in_file_path} -background nan -interpolation Cubic {spacing_str} -o {out_file_path}'
        return cmd_str_resample

    def _get_niftyreg_cmd_str(self, in_file_path, out_file_path):
        ref_str = f'-ref {self._reg_resmaple_ref}'
        flo_str = f'-flo {in_file_path}'
        res_str = f'-res {out_file_path}'
        args_str = f'-inter {self._order} -pad NaN'
        cmd_str = self._reg_resample + ' ' \
                  + ref_str + ' ' \
                  + flo_str + ' ' \
                  + res_str + ' ' \
                  + args_str

        return cmd_str

    @staticmethod
    def _get_file_list(file_list_txt):
        return read_file_contents_list(file_list_txt)


class DownSampleNiftyReg(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, ref_img, order=3):
        super().__init__(config, in_folder)
        self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder)
        self._ref_img = ScanWrapper(ref_img)
        self._order = order
        self._reg_resample = config['niftyreg_resample']
        self._reg_resmaple_ref = config['reg_resample_ref_img']

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        out_file_path = self._out_data_folder.get_file_path(idx)

        cmd_str = self._get_niftyreg_cmd_str(in_file_path, out_file_path)
        print(cmd_str, flush=True)
        os.system(cmd_str)

    def _get_niftyreg_cmd_str(self, in_file_path, out_file_path):
        ref_str = f'-ref {self._reg_resmaple_ref}'
        flo_str = f'-flo {in_file_path}'
        res_str = f'-res {out_file_path}'
        args_str = f'-inter {self._order} -pad NaN'
        cmd_str = self._reg_resample + ' ' \
                  + ref_str + ' ' \
                  + flo_str + ' ' \
                  + res_str + ' ' \
                  + args_str

        return cmd_str


class PreprocessAverageImputation(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, average_img, file_list_txt=None):
        super().__init__(config, in_folder, file_list_txt=file_list_txt)
        self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder, data_list_txt=file_list_txt)
        mkdir_p(out_folder)
        self._average_img = ScanWrapper(average_img)

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        out_file_path = self._out_data_folder.get_file_path(idx)

        in_img = ScanWrapper(in_file_path).get_data()
        average_img = self._average_img.get_data()

        np.copyto(in_img, average_img, where=(in_img != in_img))
        np.copyto(in_img, 0, where=(in_img != in_img))
        self._average_img.save_scan_same_space(out_file_path, in_img)


class ScanFolderFlatReader(AbstractParallelRoutine):
    def __init__(self, config, in_folder, ref_img):
        super().__init__(config, in_folder)
        self._ref_img = ScanWrapper(ref_img)
        self._data_matrix = []

    def read_data(self):
        print(f'Reading scan from folder {self._in_data_folder.get_folder()}')
        tic = time.perf_counter()
        self._data_matrix = self._init_data_matrix()
        self.run_non_parallel()
        # self.run_parallel()
        toc = time.perf_counter()
        print(f'Done. {toc - tic:0.4f} (s)')

    def get_data_matrix(self):
        return self._data_matrix

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        in_data = ScanWrapper(in_file_path)

        in_img = in_data.get_data()
        # self._data_matrix[idx, :] = in_img.reshape(in_data.get_number_voxel())
        self._data_matrix[idx, :] = convert_3d_2_flat(in_img)

    def _init_data_matrix(self):
        num_features = self._ref_img.get_number_voxel()
        num_sample = self.num_files()

        data_matrix = np.zeros((num_sample, num_features))
        return data_matrix


class ScanFolderBatchReader(AbstractParallelRoutine):
    def __init__(self, config, in_folder, ref_img, num_batch, file_list_txt=None):
        super().__init__(config, in_folder, file_list_txt)
        self._ref_img = ScanWrapper(ref_img)
        self._num_batch = num_batch
        self._chunk_list = self._in_data_folder.get_chunks_list(num_batch)
        self._data_matrix = []
        self._cur_idx = 0

    def read_data(self, idx_batch):
        self._reset_cur_idx()

        print(f'Reading scan from folder {self._in_data_folder.get_folder()}', flush=True)
        tic = time.perf_counter()
        cur_batch = self._chunk_list[idx_batch]
        self._init_data_matrix(len(cur_batch))
        self.run_non_parallel(cur_batch)
        toc = time.perf_counter()
        print(f'Done. {toc - tic:0.4f} (s)', flush=True)

    def get_data_matrix(self):
        return self._data_matrix

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        in_data = ScanWrapper(in_file_path)

        in_img = in_data.get_data()
        # self._data_matrix[idx, :] = in_img.reshape(in_data.get_number_voxel())
        # self._data_matrix[idx, :] = convert_3d_2_flat(in_img)
        self._data_matrix[self._cur_idx, :] = convert_3d_2_flat(in_img)
        self._cur_idx += 1

    def _init_data_matrix(self, num_sample):
        num_features = self._ref_img.get_number_voxel()
        # num_sample = self.num_files()

        del self._data_matrix
        self._data_matrix = np.zeros((num_sample, num_features))

    def _reset_cur_idx(self):
        self._cur_idx = 0
