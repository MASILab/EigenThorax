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
    parser.add_argument('--in-csv', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--x-label', type=str)
    parser.add_argument('--title', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    data_array = df[args.column_flag].to_numpy()

    fig, ax = plt.subplots()
    ax.hist(data_array, bins=np.arange(0, 6, 0.5), color='lightslategray', alpha=0.8, rwidth=0.9)
    plt.grid(axis='y', alpha=0.8)
    plt.xlabel(args.x_label)
    plt.ylabel('Count')
    plt.title(args.title)

    logger.info(f'Save image to {args.out_png}')
    plt.savefig(args.out_png, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()