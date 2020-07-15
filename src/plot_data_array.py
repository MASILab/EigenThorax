import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from tools.utils import get_logger


logger = get_logger('Plot data array')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-data-file-list', nargs='+', type=str)
    parser.add_argument('--data-label-list', nargs='+', type=str)
    parser.add_argument('--linestyle-list', nargs='+', type=str)
    parser.add_argument('--color-list', nargs='+', type=str)
    parser.add_argument('--num-feature', type=int)
    parser.add_argument('--x-label', type=str)
    parser.add_argument('--y-label', type=str)
    parser.add_argument('--out-png-path', type=str)
    args = parser.parse_args()

    num_data_file = len(args.load_data_file_list)
    load_data_file_list = args.load_data_file_list
    data_label_list = args.data_label_list
    linestyle_list = args.linestyle_list
    color_list = args.color_list

    plt.figure(figsize=(30, 15))
    font = {'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)

    x_list = range(1, args.num_feature + 1)
    ax = plt.subplot()

    for idx_data in range(num_data_file):
        # data_array = np.loadtxt(
        #     load_data_file_list[idx_data],
        #     dtype=float)[:args.num_feature]
        data_array = np.loadtxt(
            load_data_file_list[idx_data],
            dtype=float)
        y_array = np.zeros((args.num_feature,))
        y_array.fill(np.nan)
        effective_array_len = np.min(np.array([len(data_array), args.num_feature]))
        y_array[:effective_array_len] = data_array[:effective_array_len]
        label_str = data_label_list[idx_data]
        ax.plot(
            x_list,
            y_array,
            linestyle=linestyle_list[idx_data],
            linewidth=2,
            color=color_list[idx_data],
            label=label_str
        )

    ax.legend(loc='best')
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)

    logger.info(f'Save image to {args.out_png_path}')
    plt.savefig(args.out_png_path, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()