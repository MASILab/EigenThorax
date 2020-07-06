import numpy as np
import nibabel as nib
from multiprocessing import Pool
from tools.data_io import DataFolder, ScanWrapper, ScanWrapperWithMask, save_object
import os
from tools.paral import AbstractParallelRoutineSimple
from tools.utils import get_logger
from tools.pca import PCA_NII_3D


logger = get_logger('PCA analysis')


class ParalDimensionReduction(AbstractParallelRoutineSimple):
    def __init__(self, pca_bin_path, in_folder_obj, num_process):
        super().__init__(in_folder_obj, num_process)
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
            low_dim_representation = pca_nii_3d.transform(image_obj)
            result = {
                'scan_name': self._in_data_folder.get_file_name(idx),
                'low_dim': low_dim_representation
            }
            print(result)
            result_list.append(result)
        return result_list


class ParalGetResidual(AbstractParallelRoutineSimple):
    def __init__(self,
                 pca_bin_path,
                 in_folder_obj,
                 out_folder_obj,
                 mask_obj,
                 num_mode_used,
                 num_process):
        super().__init__(in_folder_obj, num_process)
        self._out_folder_obj = out_folder_obj
        self._pca_bin_path = pca_bin_path
        self._num_mode_used = num_mode_used
        self._mask_obj = mask_obj

    def _run_chunk(self, chunk_list):
        pca_nii_3d = PCA_NII_3D(None, None, 1)
        pca_nii_3d.load_pca(self._pca_bin_path)

        result_list = []
        for idx in chunk_list:
            self._in_data_folder.print_idx(idx)
            image_obj = ScanWrapperWithMask(
                self._in_data_folder.get_file_path(idx),
                self._mask_obj.get_path()
            )

            residual_data = self._get_residual_vector(
                pca_nii_3d._get_pca(),
                image_obj.get_data_flat()
            )

            out_path = self._out_folder_obj.get_file_path(idx)
            image_obj.save_scan_flat_img(residual_data, out_path)

        return result_list

    def _get_residual_vector(self, pca_obj, data_vector):
        decentered = data_vector - pca_obj.mean_
        cp_matrix = pca_obj.components_[:self._num_mode_used, :]

        mode_of_variation = np.dot(cp_matrix, decentered)
        low_dim_rep = np.dot(mode_of_variation, cp_matrix)

        residual = decentered - low_dim_rep

        return residual
