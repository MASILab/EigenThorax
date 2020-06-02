import argparse
from tools.pca import PCA_NII_3D
from tools.paral import AbstractParallelRoutineSimple
from tools.data_io import ScanWrapper, DataFolder, save_object, load_object
from tools.utils import get_logger
from tools.regression import EigenThoraxLinearRegression1D
import numpy as np


logger = get_logger('Dimension Reduction')


def main():
    parser = argparse.ArgumentParser(description='Eliminate the 1D subspace that correspond to BMI')
    parser.add_argument('--in-data-dict-bin', type=str)
    parser.add_argument('--in-feature-dim', type=int, default=20)
    parser.add_argument('--out-data-dict-bin', type=str)
    args = parser.parse_args()

    in_dict_obj = load_object(args.in_data_dict_bin)

    scan_name_list = list(in_dict_obj.keys())
    data_X = np.zeros((len(scan_name_list), args.in_feature_dim),
                      dtype=float)
    data_Y = np.zeros((len(scan_name_list),),
                      dtype=float)

    for idx_scan in range(len(scan_name_list)):
        scan_name = scan_name_list[idx_scan]
        data_X[idx_scan, :] = in_dict_obj[scan_name]['ImageData'][:]
        data_Y[idx_scan] = in_dict_obj[scan_name]['bmi']

    linear_reg_obj = EigenThoraxLinearRegression1D(data_X, data_Y)
    linear_reg_obj.run_regression()
    projected_data_X = linear_reg_obj.project_to_complement_space()

    for idx_scan in range(len(scan_name_list)):
        scan_name = scan_name_list[idx_scan]
        in_dict_obj[scan_name]['ImageData'] = projected_data_X[idx_scan, :]

    save_object(in_dict_obj, args.out_data_dict_bin)




if __name__ == '__main__':
    main()