import argparse
from tools.pca import PCA_NII_3D
from tools.data_io import ScanWrapper, DataFolder, save_object, load_object
from tools.utils import get_logger

logger = get_logger('PCA low dimension')


class DimensionReductionFromBin:
    def __init__(self, pca_bin_path, data_bin_path):
        self._pca_bin_path = pca_bin_path
        self._data_bin_path = data_bin_path

    def run_dimension_reduction(self, save_bin_path):
        pca_nii_3d = PCA_NII_3D(None, None, 1)
        pca_nii_3d.load_pca(self._pca_bin_path)

        image_feature_data_obj = load_object(self._data_bin_path)

        projected_matrix = pca_nii_3d._get_pca().transform(image_feature_data_obj['data_matrix'])

        out_data = {
            'file_list': image_feature_data_obj['file_list'],
            'projected_matrix': projected_matrix
        }

        save_object(out_data, save_bin_path)


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-pca-bin-path', type=str,
                        help='Location of the pca bin')
    parser.add_argument('--load-data-matrix-bin', type=str)
    parser.add_argument('--save-bin-path', type=str)
    args = parser.parse_args()

    dr_obj = DimensionReductionFromBin(
        args.load_pca_bin_path,
        args.load_data_matrix_bin
    )

    dr_obj.run_dimension_reduction(args.save_bin_path)


if __name__ == '__main__':
    main()