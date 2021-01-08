import os
from tools.utils import read_file_contents_list, mkdir_p, convert_3d_2_flat, get_logger
from tools.paral import AbstractParallelRoutine
from tools.data_io import DataFolder, ScanWrapper, ScanWrapperWithMask
import numpy as np
import time

logger = get_logger('Preprocess')


class PreprocessDownSample(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, ref_img, order=3):
        super().__init__(config, in_folder)
        self._c3d = config['c3d_exe']
        self._file_list = self._get_file_list(config['data_file_list'])
        self._spacing_config = config['spacing']
        self._in_folder = in_folder
        self._out_folder = out_folder
        mkdir_p(out_folder)
        self._order = order
        self._reg_resample = config['niftyreg_resample']
        self._reg_resmaple_ref = config['reg_resample_ref_img']

    def _run_chunk(self, chunk_list):
        for id_file in chunk_list:
            self._in_data_folder.print_idx(id_file)
            in_img_name = self._in_data_folder.get_file_name(id_file)
            in_img_path = self._in_data_folder.get_file_path(id_file)
            out_img_path = os.path.join(self._out_folder, in_img_name)
            c3d_cmd_str = self._get_c3d_cmd_str(in_img_path, out_img_path)
            print(c3d_cmd_str, flush=True)
            os.system(c3d_cmd_str)

    def run(self):
        for idx in range(self.num_files()):
            file_name = self._file_list[idx]
            print(f'Preprocess scan {file_name} ({idx}/{self.num_files()})', flush=True)
            self._run_single_file(file_name)
            print('Done', flush=True)

    def num_files(self):
        return len(self._file_list)

    def _run_single_file(self, file_name):
        in_file_path = os.path.join(self._in_folder, file_name)
        out_file_path = os.path.join(self._out_folder, file_name)

        # cmd_str = self._get_c3d_cmd_str(in_file_path, out_file_path)
        cmd_str = self._get_niftyreg_cmd_str(in_file_path, out_file_path)
        print(cmd_str, flush=True)
        os.system(cmd_str)

    def _get_spacing_cmd(self):
        spacing = {}
        spacing[0] = self._spacing_config['X']
        spacing[1] = self._spacing_config['Y']
        spacing[2] = self._spacing_config['Z']
        spacing_str = f'-resample-mm {spacing[0]}x{spacing[1]}x{spacing[2]}mm'

        return spacing_str

    def _get_c3d_cmd_str(self, in_file_path, out_file_path):
        spacing_str = self._get_spacing_cmd()
        cmd_str_resample = f'{self._c3d} {in_file_path} -background nan -interpolation Cubic {spacing_str} -o {out_file_path}'
        return cmd_str_resample

    def _get_niftyreg_cmd_str(self, in_file_path, out_file_path):
        ref_str = f'-ref {self._reg_resmaple_ref}'
        flo_str = f'-flo {in_file_path}'
        res_str = f'-res {out_file_path}'
        args_str = f'-inter {self._order} -pad NaN'
        cmd_str = self._reg_resample + ' ' \
                  + ref_str + ' ' \
                  + flo_str + ' ' \
                  + res_str + ' ' \
                  + args_str

        return cmd_str

    @staticmethod
    def _get_file_list(file_list_txt):
        return read_file_contents_list(file_list_txt)


class DownSampleNiftyReg(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, ref_img, order=3):
        super().__init__(config, in_folder)
        self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder)
        self._ref_img = ScanWrapper(ref_img)
        self._order = order
        self._reg_resample = config['niftyreg_resample']
        self._reg_resmaple_ref = config['reg_resample_ref_img']

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        out_file_path = self._out_data_folder.get_file_path(idx)

        cmd_str = self._get_niftyreg_cmd_str(in_file_path, out_file_path)
        print(cmd_str, flush=True)
        os.system(cmd_str)

    def _get_niftyreg_cmd_str(self, in_file_path, out_file_path):
        ref_str = f'-ref {self._reg_resmaple_ref}'
        flo_str = f'-flo {in_file_path}'
        res_str = f'-res {out_file_path}'
        args_str = f'-inter {self._order} -pad NaN'
        cmd_str = self._reg_resample + ' ' \
                  + ref_str + ' ' \
                  + flo_str + ' ' \
                  + res_str + ' ' \
                  + args_str

        return cmd_str


class PreprocessAverageImputation(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, average_img, file_list_txt=None):
        super().__init__(config, in_folder, file_list_txt=file_list_txt)
        self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder, data_list_txt=file_list_txt)
        mkdir_p(out_folder)
        self._average_img = ScanWrapper(average_img)

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        out_file_path = self._out_data_folder.get_file_path(idx)

        in_img = ScanWrapper(in_file_path).get_data()
        average_img = self._average_img.get_data()

        np.copyto(in_img, average_img, where=(in_img != in_img))
        np.copyto(in_img, 0, where=(in_img != in_img))
        self._average_img.save_scan_same_space(out_file_path, in_img)


class PreprocessClipROI(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, roi_img, file_list_txt=None):
        super().__init__(config, in_folder, file_list_txt=file_list_txt)
        # self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder, data_list_txt=file_list_txt)
        self._out_data_folder = DataFolder.get_data_folder_obj_with_list(out_folder, self._in_data_folder.get_data_file_list())
        mkdir_p(out_folder)
        self._roi_img_path = roi_img
        self._reg_resample_path = config['niftyreg_resample']

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        out_file_path = self._out_data_folder.get_file_path(idx)

        reg_resample_cmd_str = f'{self._reg_resample_path} -inter 0 -ref {self._roi_img_path} -flo {in_file_path} -res {out_file_path}'
        logger.info(reg_resample_cmd_str)
        os.system(reg_resample_cmd_str)


class ScanFolderFlatReader(AbstractParallelRoutine):
    def __init__(self, config, in_folder, ref_img):
        super().__init__(config, in_folder)
        self._ref_img = ScanWrapper(ref_img)
        self._data_matrix = []

    def read_data(self):
        print(f'Reading scan from folder {self._in_data_folder.get_folder()}')
        tic = time.perf_counter()
        self._data_matrix = self._init_data_matrix()
        self.run_non_parallel()
        # self.run_parallel()
        toc = time.perf_counter()
        print(f'Done. {toc - tic:0.4f} (s)')

    def get_data_matrix(self):
        return self._data_matrix

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        in_data = ScanWrapper(in_file_path)

        in_img = in_data.get_data()
        # self._data_matrix[idx, :] = in_img.reshape(in_data.get_number_voxel())
        self._data_matrix[idx, :] = convert_3d_2_flat(in_img)

    def _init_data_matrix(self):
        num_features = self._ref_img.get_number_voxel()
        num_sample = self.num_files()

        data_matrix = np.zeros((num_sample, num_features))
        return data_matrix


class ScanFolderBatchReader(AbstractParallelRoutine):
    def __init__(self, config, in_folder, ref_img, batch_size, file_list=None):
        super().__init__(config, in_folder, file_list)
        self._ref_img = ScanWrapper(self._in_data_folder.get_file_path(0))
        self._chunk_list = self._in_data_folder.get_chunks_list_batch_size(batch_size)
        self._data_matrix = []
        self._cur_idx = 0

    def read_data(self, idx_batch):
        self._reset_cur_idx()

        print(f'Reading scans from folder {self._in_data_folder.get_folder()}', flush=True)
        tic = time.perf_counter()
        cur_batch = self._chunk_list[idx_batch]
        self._init_data_matrix(len(cur_batch))
        self.run_non_parallel(cur_batch)
        toc = time.perf_counter()
        print(f'Done. {toc - tic:0.4f} (s)', flush=True)

    def num_batch(self):
        return len(self._chunk_list)

    def get_data_matrix(self):
        return self._data_matrix

    def get_batch_idx_list(self, idx_batch):
        return self._chunk_list[idx_batch]

    def save_flat_data(self, data_array, idx, out_folder):
        out_path_ori = os.path.join(out_folder, f'pc_{idx}.nii.gz')
        self._ref_img.save_scan_flat_img(data_array, out_path_ori)

    def get_ref(self):
        return self._ref_img

    def get_batch_file_name_list(self, idx_batch):
        batch_idx_list = self._chunk_list[idx_batch]
        file_name_list = [self._in_data_folder.get_file_name(file_idx) for file_idx in batch_idx_list]
        return file_name_list

    def _run_single_scan(self, idx):
        in_file_path = self._in_data_folder.get_file_path(idx)
        in_data = ScanWrapper(in_file_path)

        in_img = in_data.get_data()
        # self._data_matrix[idx, :] = in_img.reshape(in_data.get_number_voxel())
        # self._data_matrix[idx, :] = convert_3d_2_flat(in_img)
        self._data_matrix[self._cur_idx, :] = convert_3d_2_flat(in_img)
        self._cur_idx += 1

    def _init_data_matrix(self, num_sample):
        num_features = self._get_number_of_voxel()

        del self._data_matrix
        self._data_matrix = np.zeros((num_sample, num_features))

    def _reset_cur_idx(self):
        self._cur_idx = 0

    def _get_number_of_voxel(self):
        return self._ref_img.get_number_voxel()

class ScanFolderBatchWithMaskReader(ScanFolderBatchReader):
    def __init__(self, config, in_folder, mask_img, batch_size, file_list_txt):
        super().__init__(config, in_folder, None, batch_size, file_list_txt)
        self._mask_img = ScanWrapper(mask_img)
        self._masked_ref = ScanWrapperWithMask(
            self._in_data_folder.get_file_path(0),
            self._mask_img.get_path()
        )

    def save_flat_data(self, data_array, idx, out_folder):
        out_path = os.path.join(out_folder, f'pc_{idx}.nii.gz')

        self._masked_ref.save_scan_flat_img(data_array, out_path)

    def _run_single_scan(self, idx):
        in_data = ScanWrapperWithMask(
            self._in_data_folder.get_file_path(idx),
            self._mask_img.get_path()
        )

        self._data_matrix[self._cur_idx, :] = in_data.get_data_flat()

        self._cur_idx += 1

    def _get_number_of_voxel(self):
        return self._masked_ref.get_data_flat().shape[0]


class ScanFolderConcatBatchReader(AbstractParallelRoutine):
    def __init__(self,
                 config,
                 in_ori_folder,
                 in_jac_folder,
                 batch_size,
                 file_list_txt=None):
        super().__init__(config, in_ori_folder, file_list_txt)
        self._in_jac_folder = DataFolder(in_jac_folder, file_list_txt)
        self._ref_ori = ScanWrapper(self._in_data_folder.get_file_path(0))
        self._ref_jac = ScanWrapper(self._in_jac_folder.get_file_path(0))
        self._chunk_list = self._in_data_folder.get_chunks_list_batch_size(batch_size)
        self._data_matrix = []
        self._cur_idx = 0

    def read_data(self, idx_batch):
        self._reset_cur_idx()

        print(f'Reading scans from folder {self._in_data_folder.get_folder()}', flush=True)
        tic = time.perf_counter()
        cur_batch = self._chunk_list[idx_batch]
        self._init_data_matrix(len(cur_batch))
        self.run_non_parallel(cur_batch)
        toc = time.perf_counter()
        print(f'Done. {toc - tic:0.4f} (s)', flush=True)

    def num_batch(self):
        return len(self._chunk_list)

    def get_data_matrix(self):
        return self._data_matrix

    def save_flat_data(self, data_array, idx, out_folder):
        out_path_ori = os.path.join(out_folder, f'pc_ori_{idx}.nii.gz')
        out_path_jac = os.path.join(out_folder, f'pc_jac_{idx}.nii.gz')

        ori_data_flat = data_array[:self._ref_ori.get_number_voxel()]
        jac_data_flat = data_array[self._ref_ori.get_number_voxel():]

        self._ref_ori.save_scan_flat_img(ori_data_flat, out_path_ori)
        self._ref_jac.save_scan_flat_img(jac_data_flat, out_path_jac)

    def get_ref(self):
        return self._ref_ori

    def _run_single_scan(self, idx):
        in_ori_data = ScanWrapper(self._in_data_folder.get_file_path(idx)).get_data()
        in_jac_data = ScanWrapper(self._in_jac_folder.get_file_path(idx)).get_data()

        self._data_matrix[self._cur_idx, :self._ref_ori.get_number_voxel()] = convert_3d_2_flat(in_ori_data)
        self._data_matrix[self._cur_idx, self._ref_ori.get_number_voxel():] = convert_3d_2_flat(in_jac_data)

        self._cur_idx += 1

    def _init_data_matrix(self, num_sample):
        num_features = self._get_number_of_voxel()

        del self._data_matrix
        self._data_matrix = np.zeros((num_sample, num_features))

    def _get_number_of_voxel(self):
        return self._get_number_of_voxel_ori() + self._get_number_of_voxel_jac()

    def _get_number_of_voxel_ori(self):
        return self._ref_ori.get_number_voxel()

    def _get_number_of_voxel_jac(self):
        return self._ref_jac.get_number_voxel()

    def _reset_cur_idx(self):
        self._cur_idx = 0


class ScanFolderConcatBatchReaderWithMask(ScanFolderConcatBatchReader):
    def __init__(self,
                 mask_img_path,
                 config,
                 in_ori_folder,
                 in_jac_folder,
                 batch_size,
                 file_list_txt=None):
        super().__init__(config,
                         in_ori_folder,
                         in_jac_folder,
                         batch_size,
                         file_list_txt)
        self._mask_img = ScanWrapper(mask_img_path)
        self._masked_ref_ori = ScanWrapperWithMask(self._in_data_folder.get_file_path(0), self._mask_img.get_path())
        self._masked_ref_jac = ScanWrapperWithMask(self._in_jac_folder.get_file_path(0), self._mask_img.get_path())

    def save_flat_data(self, data_array, idx, out_folder):
        out_path_ori = os.path.join(out_folder, f'pc_ori_{idx}.nii.gz')
        out_path_jac = os.path.join(out_folder, f'pc_jac_{idx}.nii.gz')

        ori_data_flat = data_array[:self._get_number_of_voxel_ori()]
        jac_data_flat = data_array[self._get_number_of_voxel_ori():]

        self._masked_ref_ori.save_scan_flat_img(ori_data_flat, out_path_ori)
        self._masked_ref_jac.save_scan_flat_img(jac_data_flat, out_path_jac)

    def _run_single_scan(self, idx):
        in_ori_data_obj = ScanWrapperWithMask(self._in_data_folder.get_file_path(idx), self._mask_img.get_path())
        in_jac_data_obj = ScanWrapperWithMask(self._in_jac_folder.get_file_path(idx), self._mask_img.get_path())

        self._data_matrix[self._cur_idx, :self._get_number_of_voxel_ori()] = in_ori_data_obj.get_data_flat()
        self._data_matrix[self._cur_idx, self._get_number_of_voxel_ori():] = in_jac_data_obj.get_data_flat()

        self._cur_idx += 1

    def _get_number_of_voxel_ori(self):
        return self._masked_ref_ori.get_data_flat().shape[0]

    def _get_number_of_voxel_jac(self):
        return self._masked_ref_jac.get_data_flat().shape[0]


class CalcJacobian(AbstractParallelRoutine):
    def __init__(self, config, in_dat_folder, out_folder, ref_img, file_list_txt):
        super().__init__(config, in_dat_folder, file_list_txt)
        self._ref_img = ScanWrapper(ref_img)
        self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder, data_list_txt=file_list_txt)
        self._jacobian_tool = config['deedsBCV_jacobian']

    def _run_single_scan(self, idx):
        in_dat_prefix = self._in_data_folder.get_file_path(idx)
        out_file_path = self._out_data_folder.get_file_path(idx)
        ref_img_path = self._ref_img.get_path()

        jacobian_cmd_str = f'{self._jacobian_tool} -M {ref_img_path} -O {in_dat_prefix} -J {out_file_path}'
        logger.info(jacobian_cmd_str)
        os.system(jacobian_cmd_str)


class JacobianAffineCorrection(AbstractParallelRoutine):
    def __init__(self, config, in_jac_folder, in_affine_mat_folder, out_folder, ref_img, file_list_txt):
        super().__init__(config, in_jac_folder, file_list_txt)
        self._ref_img = ScanWrapper(ref_img)
        self._in_affine_mat_folder = DataFolder.get_data_folder_obj(config, in_affine_mat_folder, data_list_txt=file_list_txt)
        self._out_data_folder = DataFolder.get_data_folder_obj(config, out_folder, data_list_txt=file_list_txt)

    def _run_single_scan(self, idx):
        in_data_path = self._in_data_folder.get_file_path(idx)
        in_affine_mat_path = self._in_affine_mat_folder.get_file_path(idx).replace('.nii.gz', '.txt')
        out_file_path = self._out_data_folder.get_file_path(idx)

        in_image = ScanWrapper(in_data_path)
        affine_mat = self._get_affine_matrix(in_affine_mat_path)

        new_log_jacobian_det = in_image.get_data() + np.log(np.linalg.det(affine_mat))

        self._ref_img.save_scan_same_space(out_file_path, new_log_jacobian_det)

    def _get_affine_matrix(self, mat_path):
        return np.loadtxt(mat_path)


class NonNanRegion(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_folder, file_list_txt):
        super().__init__(config, in_folder, file_list_txt)
        self._out_folder = DataFolder.get_data_folder_obj(config, out_folder, data_list_txt=file_list_txt)

    def _run_single_scan(self, idx):
        in_img = ScanWrapper(self._in_data_folder.get_file_path(idx))
        out_mask_path = self._out_folder.get_file_path(idx)

        in_img_data = in_img.get_data()
        non_nan_mask = in_img_data == in_img_data

        logger.info(f'Save non-nan mask to {out_mask_path}')
        in_img.save_scan_same_space(out_mask_path, non_nan_mask.astype(int))


class MaskIntersection(AbstractParallelRoutine):
    def __init__(self, config, in_folder, out_img_path, file_list_txt):
        super().__init__(config, in_folder, file_list_txt)
        self._out_img_path = out_img_path
        self._ref_img = ScanWrapper(self._in_data_folder.get_first_path())

    def run(self):
        result_list = self.run_parallel()
        out_mask = self._get_intersect_mask(result_list)

        self._ref_img.save_scan_same_space(self._out_img_path, out_mask)

    def _run_single_scan(self, idx):
        in_img = ScanWrapper(self._in_data_folder.get_file_path(idx))
        return in_img.get_data()

    def _run_chunk(self, chunk_list):
        result_list = []
        inter_mask = np.ones(self._ref_img.get_shape())
        for idx in chunk_list:
            self._in_data_folder.print_idx(idx)
            mask = self._run_single_scan(idx)
            inter_mask = np.multiply(inter_mask, mask)

        result_list.append(inter_mask)
        return result_list

    def _get_intersect_mask(self, mask_list):
        inter_mask = np.ones(self._ref_img.get_shape())
        for mask in mask_list:
            inter_mask = np.multiply(inter_mask, mask)

        return inter_mask


