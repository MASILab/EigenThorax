import os
from tools.utils import read_file_contents_list, convert_flat_2_3d
import nibabel as nib
import numpy as np
import pickle


class DataFolder:
    def __init__(self, in_folder, data_file_list=None):
        self._in_folder = in_folder
        self._file_list = []
        if data_file_list is None:
            self._file_list = self._get_file_list_in_folder(in_folder)
        else:
            self._file_list = self._get_file_list(data_file_list)

    def get_folder(self):
        return self._in_folder

    def if_file_exist(self, idx):
        file_path = self.get_file_path(idx)
        return os.path.exists(file_path)

    def get_file_name(self, idx):
        return self._file_list[idx]

    def get_file_path(self, idx):
        return os.path.join(self._in_folder, self.get_file_name(idx))

    def get_first_path(self):
        return self.get_file_path(0)

    def num_files(self):
        return len(self._file_list)

    def print_idx(self, idx):
        print('Process %s (%d/%d)' % (self.get_file_path(idx), idx, self.num_files()), flush=True)

    def get_chunks_list(self, num_pieces):
        full_id_list = range(self.num_files())
        return [full_id_list[i::num_pieces] for i in range(num_pieces)]

    def get_data_file_list(self):
        return self._file_list

    @staticmethod
    def _get_file_list(file_list_txt):
        return read_file_contents_list(file_list_txt)

    @staticmethod
    def get_data_folder_obj(config, in_folder, data_list_txt=None):
        in_data_list_txt = data_list_txt
        if in_data_list_txt is None:
            in_data_list_txt = config['data_file_list']
        data_folder = DataFolder(in_folder, in_data_list_txt)
        return data_folder

    @staticmethod
    def _get_file_list_in_folder(folder_path):
        print(f'Reading file list from folder {folder_path}', flush=True)
        return os.listdir(folder_path)


class ScanWrapper:
    def __init__(self, ref_img_path):
        self._ref_img = nib.load(ref_img_path)

    def get_header(self):
        return self._ref_img.header

    def get_affine(self):
        return self._ref_img.affine

    def get_shape(self):
        return self.get_header().get_data_shape()

    def get_number_voxel(self):
        return np.prod(self.get_shape())

    def get_data(self):
        return self._ref_img.get_data()

    def save_scan_same_space(self, file_path, img_data):
        print(f'Saving image to {file_path}')
        img_obj = nib.Nifti1Image(img_data,
                                  affine=self.get_affine(),
                                  header=self.get_header())
        nib.save(img_obj, file_path)

    def save_scan_flat_img(self, data_flat, out_path):
        img_shape = self.get_shape()
        data_3d = convert_flat_2_3d(data_flat, img_shape)
        self.save_scan_same_space(out_path, data_3d)


def save_object(object_to_save, file_path):
    with open(file_path, 'wb') as output:
        print(f'Saving obj to {file_path}', flush=True)
        pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)


def load_object(file_path):
    with open(file_path, 'rb') as input_file:
        obj = pickle.load(input_file)
        return obj
