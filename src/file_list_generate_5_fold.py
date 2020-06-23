import argparse
from tools.utils import get_logger
import pandas as pd
from tools.utils import read_file_contents_list, write_list_to_file
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold


logger = get_logger('Generate 5-fold file name list.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neg-sample-list', type=str)
    parser.add_argument('--pos-sample-list', type=str)
    parser.add_argument('--out-file-list-folder', type=str)
    parser.add_argument('--n-fold', type=int, default=5)
    args = parser.parse_args()

    n_fold = KFold(n_splits=args.n_fold)

    neg_sample_list = read_file_contents_list(args.neg_sample_list)
    pos_sample_list = read_file_contents_list(args.pos_sample_list)

    n_fold_file_name_list = []

    for neg_train_idx, neg_test_idx in n_fold.split(neg_sample_list):
        neg_train_file_name_list = [neg_sample_list[idx_file_name] for idx_file_name in neg_train_idx]
        n_fold_file_name_list.append(neg_train_file_name_list)

    idx_fold = 0
    for pos_train_idx, pos_test_idx in n_fold.split(pos_sample_list):
        pos_train_file_name_list = [pos_sample_list[idx_file_name] for idx_file_name in pos_train_idx]
        train_file_name_list = n_fold_file_name_list[idx_fold] + pos_train_file_name_list
        n_fold_file_name_list[idx_fold] = train_file_name_list
        idx_fold += 1

    for idx_fold in range(args.n_fold):
        out_file_list_txt = os.path.join(args.out_file_list_folder, f'pca_fold_{idx_fold}.txt')
        write_list_to_file(n_fold_file_name_list[idx_fold], out_file_list_txt)


if __name__ == '__main__':
    main()
