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
        self._suffix = '.nii.gz'

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

    def get_chunks_list_batch_size(self, batch_size):
        num_chunks = self.num_files() // batch_size
        chunk_list = [range(batch_size*i, batch_size*(i+1)) for i in range(num_chunks)]
        if self.num_files() > num_chunks * batch_size:
            chunk_list.append(range(num_chunks * batch_size, self.num_files()))
        return chunk_list

    def get_data_file_list(self):
        return self._file_list

    def set_file_list(self, file_list):
        self._file_list = file_list

    def change_suffix(self, new_suffix):
        new_file_list = [file_name.replace(self._suffix, new_suffix) for file_name in self._file_list]
        self._file_list = new_file_list
        self._suffix = new_suffix

    @staticmethod
    def _get_file_list(file_list_txt):
        return read_file_contents_list(file_list_txt)

    @staticmethod
    def get_data_folder_obj(config, in_folder, data_list_txt=None):
        in_data_list_txt = data_list_txt
        # if in_data_list_txt is None:
        #     # in_data_list_txt = config['data_file_list']
        data_folder = DataFolder(in_folder, in_data_list_txt)
        return data_folder

    @staticmethod
    def get_data_folder_obj_with_list(in_folder, data_list):
        data_folder = DataFolder(in_folder)
        data_folder.set_file_list(data_list)
        return data_folder

    @staticmethod
    def _get_file_list_in_folder(folder_path):
        print(f'Reading file list from folder {folder_path}', flush=True)
        return os.listdir(folder_path)


class ScanWrapper:
    def __init__(self, img_path):
        self._img = nib.load(img_path)
        self._path = img_path

    def get_path(self):
        return self._path

    def get_header(self):
        return self._img.header

    def get_affine(self):
        return self._img.affine

    def get_shape(self):
        return self.get_header().get_data_shape()

    def get_number_voxel(self):
        return np.prod(self.get_shape())

    def get_data(self):
        return self._img.get_data()

    def get_center_slices(self):
        im_data = self.get_data()
        im_shape = im_data.shape
        slice_x = im_data[int(im_shape[0] / 2) - 1, :, :]
        slice_x = np.flip(slice_x, 0)
        slice_x = np.rot90(slice_x)
        slice_y = im_data[:, int(im_shape[0] / 2) - 1, :]
        slice_y = np.flip(slice_y, 0)
        slice_y = np.rot90(slice_y)
        slice_z = im_data[:, :, int(im_shape[2] / 2) - 1]
        slice_z = np.rot90(slice_z)

        return slice_x, slice_y, slice_z


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
