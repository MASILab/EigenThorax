import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from tools.utils import get_logger


logger = get_logger('Plot scree.')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-model-list', nargs='+', type=str)
    parser.add_argument('--model-name-list', nargs='+', type=str)
    parser.add_argument('--save-img-path', type=str)
    args = parser.parse_args()

    num_models = len(args.load_model_list)
    model_path_list = args.load_model_list
    model_name_list = args.model_name_list

    scree_array_list = []
    scree_cumsum_list = []

    max_len = 0
    for idx_model in range(num_models):
        logger.info(f'Loading model {model_name_list[idx_model]}')
        pca_nii_3d = PCA_NII_3D(None, None, 1)
        pca_nii_3d.load_pca(model_path_list[idx_model])
        scree_array = pca_nii_3d.get_scree_array()
        scree_cumsum = np.cumsum(scree_array)
        scree_array_list.append(scree_array)
        scree_cumsum_list.append(scree_cumsum)

        if len(scree_array) > max_len:
            max_len = len(scree_array)

        # if idx_model == 0:
        #     x = np.arange(len(scree_array)) + 1

    x = np.arange(max_len) + 1

    plt.figure(figsize=(30, 15))

    font = {'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)

    # Scree.
    ax1 = plt.subplot(1, 2, 1)
    for idx_model in range(num_models):
        y_array = np.zeros((max_len,), dtype=float)
        y_array.fill(np.nan)
        model_scree_list = scree_array_list[idx_model]
        y_array[:len(model_scree_list)] = model_scree_list

        ax1.plot(x,
                 y_array,
                 marker='+',
                 linewidth=2,
                 label=model_name_list[idx_model])

    ax1.legend(loc='best')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('% Variance Explained')
    ax1.set_title('Scree Plot')

    # Cumulative explained variance
    ax2 = plt.subplot(1, 2, 2)
    for idx_model in range(num_models):
        y_array = np.zeros((max_len,), dtype=float)
        y_array.fill(np.nan)
        model_cumsum_list = scree_cumsum_list[idx_model]
        y_array[:len(model_cumsum_list)] = model_cumsum_list

        ax2.plot(x,
                 y_array,
                 marker='+',
                 linewidth=2,
                 label=model_name_list[idx_model])

    ax2.legend(loc='best')
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Cumulative % variance explained')
    ax2.set_title('Cumulative Explained Variance')

    print(f'Plot scree to {args.save_img_path}')
    # plt.savefig(args.save_img_path, bbox_inches='tight', pad_inches=0)
    plt.savefig(args.save_img_path, bbox_inches='tight', pad_inches=1)


if __name__ == '__main__':
    main()