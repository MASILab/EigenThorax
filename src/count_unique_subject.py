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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv, index_col='Scan')
    data_dict = df.to_dict('index')

    file_list = list(data_dict.keys())

    subject_list = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name) for file_name in file_list]
    subject_list = list(set(subject_list))

    print(f'Number of subjects: {len(subject_list)}')


if __name__ == '__main__':
    main()