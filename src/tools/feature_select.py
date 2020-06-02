from tools.regression import EigenThoraxLinearRegression1D
import numpy as np
from tools.utils import get_logger
from tools.data_io import save_object, load_object


logger = get_logger('reduce_dim')


class FSDimReduction1D:
    def __init__(self, in_dict_obj, in_feature_dim):
        self._in_dict_obj = in_dict_obj
        self._feature_dim = in_feature_dim

    def run_dim_reduct(self, field_flag):
        logger.info(f'Eliminate dimension correlated to {field_flag}')
        scan_name_list = list(self._in_dict_obj.keys())
        data_X = np.zeros((len(scan_name_list), self._feature_dim),
                          dtype=float)
        data_Y = np.zeros((len(scan_name_list),),
                          dtype=float)

        for idx_scan in range(len(scan_name_list)):
            scan_name = scan_name_list[idx_scan]
            data_X[idx_scan, :] = self._in_dict_obj[scan_name]['ImageData'][:]
            data_Y[idx_scan] = self._in_dict_obj[scan_name][field_flag]

        linear_reg_obj = EigenThoraxLinearRegression1D(data_X, data_Y)
        linear_reg_obj.run_regression()
        projected_data_X = linear_reg_obj.project_to_complement_space()

        for idx_scan in range(len(scan_name_list)):
            scan_name = scan_name_list[idx_scan]
            self._in_dict_obj[scan_name]['ImageData'] = projected_data_X[idx_scan, :]

        self._feature_dim -= 1

    def save_bin(self, out_path):
        save_object(self._in_dict_obj, out_path)

