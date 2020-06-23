import argparse
from tools.data_io import load_object
from tools.utils import get_logger
import os
from tools.ica import RunICA


logger = get_logger('ICA')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--in-data-dict-bin', type=str)
    parser.add_argument('--out-data-dict-bin', type=str)
    args = parser.parse_args()

    in_data_dict = load_object(args.in_data_dict_bin)
    ica_obj = RunICA(in_data_dict)

    ica_obj.run_ica()
    ica_obj.save_data_dict_bin(args.out_data_dict_bin)


if __name__ == '__main__':
    main()