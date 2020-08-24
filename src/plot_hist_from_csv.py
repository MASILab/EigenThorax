import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from tools.utils import get_logger
import pandas as pd


logger = get_logger('Bin Plot')


def main():
    parser = argparse.ArgumentParser(description='Bin plot data using csv file')
    parser.add_argument('--in-csv-list', nargs='+', type=str)
    parser.add_argument('--label-list', nargs='+', type=str)
    parser.add_argument('--color-list', nargs='+', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--x-label', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    num_data = len(args.in_csv_list)
    in_csv_list = args.in_csv_list
    label_list = args.label_list
    color_list = args.color_list

    data_array_sequence = []
    for idx in range(num_data):
        df = pd.read_csv(in_csv_list[idx])
        data_array = df[args.column_flag].to_numpy()
        data_array_sequence.append(data_array)

    fig, ax = plt.subplots()
    ax.hist(
        data_array_sequence,
        bins='auto',
        color=color_list, label=label_list, alpha=0.8, rwidth=0.9)
    ax.legend(loc='best')
    plt.grid(axis='y', alpha=0.8)
    plt.xlabel(args.x_label)
    plt.ylabel('Count')
    plt.title(args.title)

    logger.info(f'Save image to {args.out_png}')
    plt.savefig(args.out_png, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()