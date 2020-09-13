import argparse
from tools.data_io import save_object, load_object
from tools.utils import get_logger, read_file_contents_list
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools.clinical import ClinicalDataReaderSPORE
import math


logger = get_logger('Pairwise distance')


def check_if_same_subject(file_name1, file_name2):
    subject1_id = ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name1)
    subject2_id = ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name2)

    return subject1_id == subject2_id

def get_idx_pair_of_intra_inter_groups(file_list):
    intra_pair_list = []
    inter_pair_list = []

    for idx1 in range(len(file_list)):
        file_name1 = file_list[idx1]
        for idx2 in range(idx1 + 1, len(file_list)):
            file_name2 = file_list[idx2]
            idx_pair = [idx1, idx2]
            if check_if_same_subject(file_name1, file_name2):
                intra_pair_list.append(idx_pair)
            else:
                inter_pair_list.append(idx_pair)

    logger.info(f'Num of intra-subject pair: {len(intra_pair_list)}')
    logger.info(f'Num of inter-subject pair: {len(inter_pair_list)}')

    # logger.info('List of intra subject pairs')
    # print(intra_pair_list)

    return intra_pair_list, inter_pair_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv', type=str)
    parser.add_argument('--metric-column', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv, index_col='Scan')
    data_dict = df.to_dict('index')

    file_list = list(data_dict.keys())
    value_list = np.array([data_dict[file_name][args.metric_column] for file_name in file_list])

    intra_pair_list, inter_pair_list = get_idx_pair_of_intra_inter_groups(file_list)

    intra_pair_dist_list = np.zeros((len(intra_pair_list),), dtype=float)
    inter_pair_dist_list = np.zeros((len(inter_pair_list),), dtype=float)

    for idx_pair in range(len(intra_pair_list)):
        pair = intra_pair_list[idx_pair]
        intra_pair_dist_list[idx_pair] = math.fabs(value_list[pair[0]] - value_list[pair[1]])

    for idx_pair in range(len(inter_pair_list)):
        pair = inter_pair_list[idx_pair]
        inter_pair_dist_list[idx_pair] = math.fabs(value_list[pair[0]] - value_list[pair[1]])

    logger.info(f'Inter dist mean: {np.mean(inter_pair_dist_list)}')
    logger.info(f'Intra dist mean: {np.mean(intra_pair_dist_list)}')
    logger.info(f'Ratio: {np.mean(inter_pair_dist_list) / np.mean(intra_pair_dist_list)}')

    data_array_sequence = [inter_pair_dist_list, intra_pair_dist_list]
    fig, ax = plt.subplots(figsize=(18, 12))
    color_list = ['red', 'blue']
    label_list = ['Inter-subject', 'Intra-subject']

    hist_info = ax.hist(
        data_array_sequence,
        bins=10,
        color=color_list,
        label=label_list,
        alpha=0.5,
        rwidth=0.8
    )
    # print(hist_info)
    ax.legend(loc='best')

    ax.set_ylabel('Count')
    ax.set_xlabel('Mahalanobis distance')

    logger.info(f'Save plot to {args.out_png}')
    plt.savefig(args.out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()






if __name__ == '__main__':
    main()