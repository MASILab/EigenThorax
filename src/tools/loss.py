from tools.paral import AbstractParallelRoutine
from tools.data_io import DataFolder, ScanWrapper
import os
from skimage.measure import compare_nrmse
import numpy as np


class GetLossBetweenFolder(AbstractParallelRoutine):
    def __init__(self, config, in_folder_1, in_folder_2, file_list_txt):
        super().__init__(config, in_folder_1, file_list_txt)
        self._in_data_folder_2 = DataFolder(in_folder_2, file_list_txt)
        self._nrmse_diff = []

    def get_nrmse(self):
        return self._nrmse_diff

    def print_file_list(self):
        file_list = self._in_data_folder.get_data_file_list()
        for idx in range(len(file_list)):
            print(f'The {idx}th file is {file_list[idx]}')

    def _run_single_scan(self, idx):
        in_file_1_path = self._in_data_folder.get_file_path(idx)
        in_file_2_path = self._in_data_folder_2.get_file_path(idx)

        in_img_1 = ScanWrapper(in_file_1_path).get_data()
        in_img_2 = ScanWrapper(in_file_2_path).get_data()

        nrmse = compare_nrmse(np.abs(in_img_1), np.abs(in_img_2))
        self._nrmse_diff.append(nrmse)

