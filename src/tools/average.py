import numpy as np
import nibabel as nib
from multiprocessing import Pool
from tools.data_io import DataFolder, ScanWrapper
import os
from tools.paral import AbstractParallelRoutine
from tools.utils import get_logger

logger = get_logger('Average')


class AverageScans:
    def __init__(self, config, in_folder=None, data_file_txt=None, in_data_folder_obj=None):
        self._data_folder = None
        if in_data_folder_obj is None:
            self._data_folder = DataFolder(in_folder, data_file_txt)
        else:
            self._data_folder = in_data_folder_obj
        self._standard_ref = ScanWrapper(self._data_folder.get_first_path())
        self._num_processes = config['num_processes']

    def get_average_image_union(self, save_path):
        im_shape = self._get_std_shape()

        average_union = np.zeros(im_shape)
        average_union.fill(np.nan)
        non_null_mask_count_image = np.zeros(im_shape)

        chunk_list = self._data_folder.get_chunks_list(self._num_processes)

        pool = Pool(processes=self._num_processes)

        print('Average in union')
        print('Step.1 Summation')
        image_average_union_result_list = [pool.apply_async(self._sum_images_union,
                                                            (file_idx_chunk,))
                                           for file_idx_chunk in chunk_list]
        for thread_idx in range(len(image_average_union_result_list)):
            result = image_average_union_result_list[thread_idx]
            result.wait()
            print(f'Thread with idx {thread_idx} / {len(image_average_union_result_list)} is completed')
            print('Adding to averaged_image...')
            averaged_image_chunk = result.get()
            average_union = self._add_image_union(average_union, averaged_image_chunk)
            print('Done.')

        print('Step.2 Non-nan counter')
        non_null_mask_count_result = [pool.apply_async(self._sum_non_null_count,
                                                       (file_idx_chunk,))
                                      for file_idx_chunk in chunk_list]
        for thread_idx in range(len(non_null_mask_count_result)):
            result = non_null_mask_count_result[thread_idx]
            result.wait()
            print(f'Thread with idx {thread_idx} / {len(non_null_mask_count_result)} is completed')
            print('Adding to averaged_image...')
            averaged_image_chunk = result.get()
            non_null_mask_count_image = np.add(non_null_mask_count_image, averaged_image_chunk)
            print('Done.')

        average_union = np.divide(average_union,
                                  non_null_mask_count_image,
                                  out=average_union,
                                  where=non_null_mask_count_image > 0)

        self._standard_ref.save_scan_same_space(save_path, average_union)
        print('Done.')

    def _sum_images_union(self, chunk_list):
        print('Sum images, union non-null region. Loading images...')
        im_shape = self._get_std_shape()
        sum_image = np.zeros(im_shape)
        sum_image.fill(np.nan)

        for id_file in chunk_list:
            file_path = self._data_folder.get_file_path(id_file)
            self._data_folder.print_idx(id_file)

            im = nib.load(file_path)
            im_data = im.get_data()
            sum_image = self._add_image_union(sum_image, im_data)

        return sum_image

    def _sum_non_null_count(self, chunk_list):
        print('Count non-null per voxel. Loading images...')
        im_shape = self._get_std_shape()
        sum_image = np.zeros(im_shape)

        for id_file in chunk_list:
            file_path = self._data_folder.get_file_path(id_file)
            self._data_folder.print_idx(id_file)
            im = nib.load(file_path)
            im_data = im.get_data()

            sum_image = np.add(sum_image, 1, out=sum_image, where=np.logical_not(np.isnan(im_data)))

        return sum_image

    def _get_std_shape(self):
        return self._standard_ref.get_data().shape

    @staticmethod
    def _add_image_inter(image1, image2):
        return np.add(image1, image2,
                      out=np.full_like(image1, np.nan),
                      where=np.logical_not(np.logical_or(np.isnan(image1),
                                                         np.isnan(image2))))

    @staticmethod
    def _add_image_union(image1, image2):
        add_image = np.full_like(image1, np.nan)
        add_image[np.logical_not(np.logical_and(np.isnan(image1),
                                                np.isnan(image2)))] = 0

        add_image = np.add(add_image, image1, out=add_image, where=np.logical_not(np.isnan(image1)))
        add_image = np.add(add_image, image2, out=add_image, where=np.logical_not(np.isnan(image2)))

        return add_image

    @staticmethod
    def sum_non_null_count(file_list, in_folder):
        print('Count non-null per voxel. Loading images...')
        im_temp = nib.load(os.path.join(in_folder, file_list[0]))
        im_temp_data = im_temp.get_data()

        sum_image = np.zeros_like(im_temp_data)

        for id_file in range(len(file_list)):
            file_name = file_list[id_file]
            print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
            file_path = os.path.join(in_folder, file_name)
            im = nib.load(file_path)
            im_data = im.get_data()

            sum_image = np.add(sum_image, 1, out=sum_image, where=np.logical_not(np.isnan(im_data)))

        return sum_image


class AverageValidRegion(AbstractParallelRoutine):
    def __init__(self, config, in_scan_folder, in_region_mask_folder, file_list_txt, out_average_img, ambient=-5000):
        super().__init__(config, in_scan_folder, file_list_txt)
        self._in_region_mask_folder = DataFolder.get_data_folder_obj(config, in_region_mask_folder,
                                                                     data_list_txt=file_list_txt)
        self._out_img_path = out_average_img
        self._ref_img = ScanWrapper(self._in_data_folder.get_first_path())
        self._ambient = ambient

    def run_get_average(self):
        result_list = self.run_parallel()

        im_shape = self._ref_img.get_shape()
        sum_image = np.zeros(im_shape)
        region_count = np.zeros(im_shape)

        for result in result_list:
            sum_image += result['sum_image']
            region_count += result['region_count']

        sum_image = np.divide(
            sum_image,
            region_count,
            out=sum_image,
            where=region_count > 0.5
        )

        sum_image_ma = np.ma.masked_array(sum_image, mask=region_count == 0)

        self._ref_img.save_scan_same_space(self._out_img_path, sum_image_ma.filled(self._ambient))
        self._ref_img.save_scan_same_space(self._out_img_path + '_region_count.nii.gz', region_count)

    def _run_chunk(self, chunk_list):
        result_list = []
        im_shape = self._ref_img.get_shape()
        sum_image_union = np.zeros(im_shape)
        region_mask_count_image = np.zeros(im_shape)
        for idx in chunk_list:
            self._in_data_folder.print_idx(idx)
            img_obj = ScanWrapper(self._in_data_folder.get_file_path(idx))
            mask_obj = ScanWrapper(self._in_region_mask_folder.get_file_path(idx))
            sum_image_union += img_obj.get_data()
            region_mask_count_image += mask_obj.get_data()

        result = {
            'sum_image': sum_image_union,
            'region_count': region_mask_count_image
        }

        result_list.append(result)
        return result_list
