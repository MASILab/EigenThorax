import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator



def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-pca-bin-path', type=str,
                        help='Location of the pca bin')
    parser.add_argument('--save-img-path', type=str)
    args = parser.parse_args()

    pca_nii_3d = PCA_NII_3D(None, None, 1)
    pca_nii_3d.load_pca(args.load_pca_bin_path)

    scree_array = pca_nii_3d.get_scree_array()
    scree_cumsum = np.cumsum(scree_array)

    print('Scree array:', flush=True)
    print(scree_array, flush=True)
    print('Cumulative variance:', flush=True)
    print(scree_cumsum, flush=True)


    x = np.arange(len(scree_array)) + 1
    # x = range(20)
    # scree_array = x
    # scree_cumsum = x

    plt.figure(figsize=(30, 15))

    font = {'weight': 'bold',
            'size': 22}

    matplotlib.rc('font', **font)

    plt.subplot(1, 2, 1)
    plt.plot(x, scree_array, 'ro-', linewidth=2)
    plt.xticks(x[::10] - 1)
    plt.xlabel('PCs')
    # plt.ylabel('Explained variance')
    plt.title('Scree plot')

    plt.subplot(1, 2, 2)
    plt.plot(x, scree_cumsum, 'bo-', linewidth=2)
    plt.xlabel('PCs')
    plt.xticks(x[::10] -1)
    # plt.ylabel('Cumulative explained variance')
    plt.title('Cumulative explained variance')

    print(f'Plot scree to {args.save_img_path}')
    # plt.savefig(args.save_img_path, bbox_inches='tight', pad_inches=0)
    plt.savefig(args.save_img_path, bbox_inches='tight', pad_inches=1)


if __name__ == '__main__':
    main()