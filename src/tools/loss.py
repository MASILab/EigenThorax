from tools.paral import AbstractParallelRoutine
from tools.data_io import DataFolder, ScanWrapper
import os
from skimage.measure import compare_nrmse
import numpy as np
import pandas as pd
from skimage import metrics
import subprocess
import re
from tools.utils import get_logger

logger = get_logger('Loss')

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


class CalculateMSE(AbstractParallelRoutine):
    def __init__(self, config, in_folder, ref_img, file_list_txt):
        super().__init__(config, in_folder, file_list_txt)
        self._ref_img = ScanWrapper(ref_img)
        self._df_mse_table = None
        self._niftyreg_reg_measure_path = config['niftyreg_reg_measure']

    def calculate_mse_array(self):
        logger.info('Calculate MSE')
        result_list = self.run_parallel()
        self._df_mse_table = pd.DataFrame(result_list)
        print(self._df_mse_table)
        logger.info('Done')

    def save_mse_csv(self, csv_file):
        logger.info(f'Save csv to {csv_file}')
        self._df_mse_table.to_csv(csv_file, mode='w', index=False)

    def get_minimal_mse_scan(self, N=1):
        pass

    def _run_single_scan(self, idx):
        in_data_path = self._in_data_folder.get_file_path(idx)

        in_img = ScanWrapper(in_data_path)
        mse_val = self._calc_nmse(self._ref_img, in_img)
        nmi_val = self._calc_nmi(self._niftyreg_reg_measure_path,
                                 self._ref_img, in_img)

        val = {
            'Scan': self._in_data_folder.get_file_name(idx),
            'MSE': mse_val,
            'NMI': nmi_val
        }

        print(val)
        return val

    @staticmethod
    def _calc_nmse(ref_img: ScanWrapper, in_img: ScanWrapper):
        mse_val = metrics.normalized_root_mse(
            image_true=ref_img.get_data(),
            image_test=in_img.get_data()
        )
        return mse_val

    @staticmethod
    def _calc_nmi(niftyreg_reg_measure, ref_img: ScanWrapper, in_img: ScanWrapper):
        cmd_str = f'{niftyreg_reg_measure} -ref {ref_img.get_path()} -flo {in_img.get_path()} -nmi'
        # logger.info(cmd_str)
        result = subprocess.check_output(cmd_str, shell=True)
        result_str = result.decode("utf-8")
        match_list = re.match(r"NMI: (?P<nmi_val>\d+\.\d+)", result_str)
        nmi_val = float(match_list.group('nmi_val'))
        return nmi_val