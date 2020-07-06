import os
from tools.utils import read_file_contents_list, convert_flat_2_3d, get_logger
import nibabel as nib
import numpy as np
import pickle
import pandas as pd
import seaborn as sns


logger = get_logger('DataFolder')


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
        logger.info('Process %s (%d/%d)' % (self.get_file_path(idx), idx, self.num_files()))

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


class ScanWrapperWithMask(ScanWrapper):
    def __init__(self, img_path, mask_path):
        super().__init__(img_path)
        self._mask = nib.load(mask_path)
        self._mask_path = mask_path

    def get_number_voxel(self):
        mask_data = self._mask.get_data()
        num_effective_voxel = np.sum(mask_data.flatten())

        return num_effective_voxel

    def get_data_flat(self):
        img_data = self.get_data()
        mask_data = self._mask.get_data()
        img_data_flat = img_data[mask_data == 1]

        return img_data_flat

    def save_scan_flat_img(self, data_flat, out_path):
        mask_data = self._mask.get_data()
        img_data = np.zeros(mask_data.shape, dtype=float)

        img_data[mask_data == 1] = data_flat

        self.save_scan_same_space(out_path, img_data)


class ClusterAnalysisDataDict:
    def __init__(self, data_dict, n_feature):
        self._data_dict = data_dict
        self._n_feature = n_feature
        self._df_field = self.get_df_field()

    def save_bin_plot_field(self, field_flag, out_png_folder):
        field_type = self._get_field_type(field_flag)
        if field_type == 'Category':
            self._save_bin_plot_field_category(field_flag)

    def _save_bin_plot_field_category(self, field_flag):
        sns.set(style="darkgrid")
        ax = sns.countplot(x=field_flag, data=self._df_field)

    def get_df_field(self):
        df_field_ori = self._get_dataframe_from_data_dict()
        df_field = df_field_ori[df_field_ori['CancerSubjectFirstScan'] != 0]
        df_field = self._modify_df_field_value(df_field, ['CancerSubjectFirstScan', 'CAC', 'COPD'])
        return df_field

    def get_features_and_labels(self, field_flag):
        df_field = self._df_field
        n_sample = df_field.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)
        for feature_idx in range(self._n_feature):
            pc_str = self._get_pc_str(feature_idx)
            data_X[:, feature_idx] = df_field[pc_str].tolist()[:]
        data_Y = df_field[field_flag].tolist()
        data_Y = np.array(data_Y)

        return data_X, data_Y

    def get_num_feature(self):
        return self._n_feature

    def get_num_sample(self):
        return len(self._data_dict)

    def _get_dataframe_from_data_dict(self):
        new_data_dict = {}
        count_cancer = 0
        count_first_cancer = 0
        for scan_name in self._data_dict:
            new_data_item = self._data_dict[scan_name]
            image_data = new_data_item.pop('ImageData')
            for idx_image_feature in range(self._n_feature):
                pc_name_str = self._get_pc_str(idx_image_feature)
                new_data_item[pc_name_str] = image_data[idx_image_feature]
            new_data_dict[scan_name] = new_data_item
            if 'CancerSubjectFirstScan' in new_data_item:
                if new_data_item['CancerSubjectFirstScan'] == 1:
                    count_first_cancer += 1
            if "Cancer" in new_data_item:
                if new_data_item['Cancer'] == 1:
                    count_cancer += 1
            if new_data_item['Packyear'] > 300:
                new_data_item['Packyear'] = 51.34
        logger.info(f'Count first cancer: {count_first_cancer}')
        logger.info(f'Count cancer: {count_cancer}')
        df = pd.DataFrame.from_dict(new_data_dict, orient='index')
        return df

    def _modify_df_field_value(self, ori_df, field_flag_list):
        for field_flag in field_flag_list:
            ori_df = self._modify_df_one_field(ori_df, field_flag)

        return ori_df

    def _modify_df_one_field(self, ori_df, field_flag):
        if (field_flag == 'copd') | (field_flag == 'COPD'):
            df_field = ori_df.fillna(value={field_flag: 2})
            df_field = df_field.replace({field_flag: {'Yes': 1, 'No': 0}})
        elif field_flag == 'Age':
            df_field = ori_df
            df_field.loc[ori_df[field_flag] < 60, field_flag] = 0
            df_field.loc[(ori_df[field_flag] < 70) & (ori_df[field_flag] >= 60), field_flag] = 1
            df_field.loc[ori_df[field_flag] >= 70, field_flag] = 2
        elif (field_flag == 'packyearsreported') | (field_flag == 'Packyear'):
            df_field = ori_df

            df_field.loc[ori_df[field_flag] < 35, field_flag] = 0
            df_field.loc[(ori_df[field_flag] >= 35) & (ori_df[field_flag] < 60), field_flag] = 1
            df_field.loc[ori_df[field_flag] >= 60, field_flag] = 2

        elif (field_flag == 'Coronary Artery Calcification') | (field_flag == 'CAC'):
            df_field = ori_df
            df_field.loc[df_field[field_flag] == 'None', field_flag] = 0
            df_field.loc[df_field[field_flag] == 'Mild', field_flag] = 1
            df_field.loc[df_field[field_flag] == 'Moderate', field_flag] = 2
            df_field.loc[df_field[field_flag] == 'Severe', field_flag] = 3

            df_field = df_field.replace(
                {field_flag:
                     {'Severe': 3, 'Moderate': 2, 'Mild': 1, 'None': 0}}
            )

        elif field_flag == 'bmi':
            df_field = ori_df
            df_field.loc[ori_df[field_flag] < 21, field_flag] = 0
            df_field.loc[(ori_df[field_flag] >= 21) & (ori_df[field_flag] < 35), field_flag] = 1
            df_field.loc[ori_df[field_flag] >= 35, field_flag] = 2
        elif (field_flag == 'cancer_bengin') | (field_flag == 'Cancer'):
            df_field = ori_df
        elif field_flag == 'CancerIncubation':
            df_field = ori_df[ori_df[field_flag] != 0]
            df_field = df_field.fillna(value={field_flag: 0})
        elif field_flag == 'CancerSubjectFirstScan':
            df_field = ori_df[ori_df[field_flag] != 0]
            df_field = df_field.fillna(value={field_flag: 0})
        else:
            raise NotImplementedError

        return df_field


    def _get_field_label_list(self, field_flag):
        label_list = []
        if (field_flag == 'copd') | (field_flag == 'COPD'):
            label_list.append(f'copd:no ')
            label_list.append(f'copd:yes ')
            label_list.append(f'copd:unknown ')
        elif field_flag == 'Age':
            label_list.append(f'Age<60 ')
            label_list.append(f'60<=Age<70 ')
            label_list.append(f'Age>=70 ')
        elif (field_flag == 'packyearsreported') | (field_flag == 'Packyear'):
            label_list.append(f'packyear<35 ')
            label_list.append(f'35<=packyear<60 ')
            label_list.append(f'packyear>=60 ')
        elif (field_flag == 'Coronary Artery Calcification') | (field_flag == 'CAC'):
            label_list.append(f'CAC: None ')
            label_list.append(f'CAC: Mild ')
            label_list.append(f'CAC: Moderate ')
            label_list.append(f'CAC: Severe ')
        elif field_flag == 'bmi':
            label_list.append(f'BMI < 21 ')
            label_list.append(f'21 <= BMI < 35 ')
            label_list.append(f'BMI >= 35 ')
        elif (field_flag == 'cancer_bengin') | (field_flag == 'Cancer'):
            label_list.append(f'non-cancer ')
            label_list.append(f'cancer ')
        elif field_flag == 'CancerIncubation':
            label_list.append(f'non-cancer ')
            label_list.append(f'cancer, time to diag >= 1y ')
        elif field_flag == 'CancerSubjectFirstScan':
            label_list.append(f'non-cancer ')
            label_list.append(f'cancer, first scan ')
        else:
            raise NotImplementedError

        return label_list

    @staticmethod
    def _get_field_type(field_flag):
        field_type = 'Unknown'
        if (field_flag == 'copd') | (field_flag == 'COPD') | (field_flag == 'Coronary Artery Calcification') | (field_flag == 'CAC'):
            field_type = 'Category'
        else:
            field_type = 'Continuous'

        return field_type

    @staticmethod
    def _get_pc_str(idx):
        return f'pc{idx}'


def save_object(object_to_save, file_path):
    with open(file_path, 'wb') as output:
        print(f'Saving obj to {file_path}', flush=True)
        pickle.dump(object_to_save, output, pickle.HIGHEST_PROTOCOL)


def load_object(file_path):
    with open(file_path, 'rb') as input_file:
        obj = pickle.load(input_file)
        return obj
