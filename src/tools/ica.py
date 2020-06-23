from sklearn.decomposition import FastICA
from tools.data_io import save_object, load_object
import numpy as np


class RunICA:
    def __init__(self, pca_data_dict):
        self._data_dict = pca_data_dict
        first_key = next(iter(self._data_dict))
        self._n_features = len(self._data_dict[first_key]['ImageData'])
        self._n_samples = len(self._data_dict)

    def run_ica(self):
        data_X = np.zeros((self._n_samples, self._n_features), dtype=float)

        file_name_list = list(self._data_dict.keys())
        for idx_sample in range(self._n_samples):
            file_name = file_name_list[idx_sample]
            data_X[idx_sample, :] = self._data_dict[file_name]['ImageData'][:]

        transformer = FastICA(n_components=self._n_features)
        data_X_transformed = transformer.fit_transform(data_X)

        for idx_sample in range(self._n_samples):
            file_name = file_name_list[idx_sample]
            self._data_dict[file_name]['ImageData'] = data_X_transformed[idx_sample, :]

    def save_data_dict_bin(self, bin_file_path):
        save_object(self._data_dict, bin_file_path)




