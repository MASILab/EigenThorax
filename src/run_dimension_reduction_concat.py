import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from tools.paral import AbstractParallelRoutineSimple
from tools.data_io import ScanWrapper, DataFolder, save_object
from tools.utils import get_logger

logger = get_logger('PCA low dimension')


class ParalDimensionReductionConcat(AbstractParallelRoutineSimple):
    def __init__(self,
                 pca_bin_path,
                 in_ori_folder_obj,
                 in_jac_folder_obj,
                 num_process):
        super().__init__(in_ori_folder_obj, num_process)
        self._in_jac_folder_obj = in_jac_folder_obj
        self._pca_bin_path = pca_bin_path
        self._low_dim_dict = {}

    def run_dimension_reduction(self, save_bin_path):
        self._low_dim_dict = self.run_parallel()
        save_object(self._low_dim_dict, save_bin_path)

    def _run_chunk(self, chunk_list):
        pca_nii_3d = PCA_NII_3D(None, None, 1)
        pca_nii_3d.load_pca(self._pca_bin_path)

        result_list = []
        for idx in chunk_list:
            self._in_data_folder.print_idx(idx)
            image_obj = ScanWrapper(self._in_data_folder.get_file_path(idx))
            jac_obj = ScanWrapper(self._in_jac_folder_obj.get_file_path(idx))
            low_dim_representation = pca_nii_3d.transform_concat(image_obj, jac_obj)
            result = {
                'scan_name': self._in_data_folder.get_file_name(idx),
                'low_dim': low_dim_representation
            }
            print(result)
            result_list.append(result)
        return result_list

def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-pca-bin-path', type=str,
                        help='Location of the pca bin')
    parser.add_argument('--in-ori-folder', type=str)
    parser.add_argument('--in-jac-folder', type=str)
    parser.add_argument('--file-list-txt', type=str)
    parser.add_argument('--save-bin-path', type=str)
    parser.add_argument('--num-process', type=int, default=10)
    args = parser.parse_args()

    in_ori_folder_obj = DataFolder(args.in_ori_folder, args.file_list_txt)
    in_jac_folder_obj = DataFolder(args.in_jac_folder, args.file_list_txt)

    paral_obj = ParalDimensionReductionConcat(
        args.load_pca_bin_path,
        in_ori_folder_obj,
        in_jac_folder_obj,
        args.num_process)

    paral_obj.run_dimension_reduction(args.save_bin_path)


if __name__ == '__main__':
    main()