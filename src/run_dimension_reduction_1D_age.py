import argparse
from tools.data_io import ScanWrapper, DataFolder, save_object, load_object
from tools.utils import get_logger
from tools.feature_select import FSDimReduction1D
import numpy as np


logger = get_logger('Dimension Reduction')


def main():
    parser = argparse.ArgumentParser(description='Eliminate the 1D subspace that correspond to BMI')
    parser.add_argument('--in-data-dict-bin', type=str)
    parser.add_argument('--in-feature-dim', type=int, default=20)
    parser.add_argument('--out-data-dict-bin', type=str)
    args = parser.parse_args()

    in_dict_obj = load_object(args.in_data_dict_bin)
    fs_obj = FSDimReduction1D(in_dict_obj, args.in_feature_dim)
    fs_obj.run_dim_reduct('Age')
    fs_obj.save_bin(args.out_data_dict_bin)


if __name__ == '__main__':
    main()