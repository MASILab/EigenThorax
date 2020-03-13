import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-pca-bin-path', type=str,
                        help='Location of the pca bin')
    parser.add_argument('--save-img-path', type=str)
    args = parser.parse_args()

    pca_nii_3d = PCA_NII_3D(None, None, 1)
    pca_nii_3d.load_pca(args.load_pca_bin_path)

    scree_array = pca_nii_3d.get_scree_array()

    x = np.arange(len(scree_array)) + 1
    plt.plot(x, scree_array, 'ro-', linewidth=2)
    plt.xlabel('PCs')
    plt.ylabel('Variance')
    plt.title('Scree plot')

    print(f'Plot scree to {args.save_img_path}')
    plt.savefig(args.save_img_path, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main()