import argparse
from tools.data_io import save_object, load_object
from tools.utils import get_logger, read_file_contents_list
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


logger = get_logger('Gaussian fit')


class FitGaussianJoint:
    def __init__(self):
        self._in_res_matrix_obj = None
        self._in_jac_matrix_obj = None
        self._num_res_pc = None
        self._num_jac_pc = None
        self._file_list = None
        self._use_data_matrix = None
        self._gaussian_model = None

    def load_data(
            self,
            in_res_matrix_path,
            num_res_pc,
            in_jac_matrix_path,
            num_jac_pc
    ):
        self._in_res_matrix_obj = load_object(in_res_matrix_path)
        self._num_res_pc = num_res_pc
        self._in_jac_matrix_obj = load_object(in_jac_matrix_path)
        self._num_jac_pc = num_jac_pc

        self._file_list = self._in_res_matrix_obj['file_list']

        num_dim = num_res_pc + num_jac_pc
        num_sample = self._in_res_matrix_obj['projected_matrix'].shape[0]
        self._use_data_matrix = np.zeros((num_sample, num_dim))
        self._use_data_matrix[:, :self._num_res_pc] = self._in_res_matrix_obj['projected_matrix'][:, :self._num_res_pc]
        self._use_data_matrix[:, self._num_res_pc:num_dim] = self._in_jac_matrix_obj['projected_matrix'][:, :self._num_jac_pc]

    def fit_gaussian(self):
        """
        Get the gaussian model ready.
        :return:
        """
        num_sample = self._use_data_matrix.shape[0]
        logger.info(f'Number of samples {num_sample}')
        logger.info(f'MLE with Multivariate Gaussian')
        mu = np.average(self._use_data_matrix, axis=0)
        x_minus_mu = self._use_data_matrix - mu
        cov_mat = np.dot(np.transpose(x_minus_mu), x_minus_mu) / num_sample

        self._gaussian_model = multivariate_normal(mean=mu, cov=cov_mat)
        logger.info(f'Done')

        print(mu)
        print(cov_mat)

        return mu, cov_mat

    def save_gaussian_model(
            self,
            out_bin
    ):
        save_object(self._gaussian_model, out_bin)

    def get_distribution(
            self,
            file_list,
            out_csv
    ):
        df = self.get_distribution_file_list(file_list)
        logger.info(f'Save Probability array to {out_csv}')
        df.to_csv(out_csv, index=False)

    def get_distribution_all(
            self,
            out_csv
    ):
        df = self.get_distribution_file_list(self._file_list)
        logger.info(f'Save distribution of all files to {out_csv}')
        df.to_csv(out_csv, index=False)

    def get_distribution_file_list(
            self,
            file_list
    ):
        mu, cov = self.fit_gaussian()
        effect_file_array = []
        p_array = []
        m_dist_array = []

        logger.info(f'Get probability for each cancer patient')
        for scan_name in file_list:
            if scan_name in self._file_list:
                effect_file_array.append(scan_name)
                scan_idx = self._file_list.index(scan_name)
                scan_x = self._use_data_matrix[scan_idx]
                scan_p = multivariate_normal.pdf(scan_x, mean=mu, cov=cov)
                scan_m_dist = mahalanobis(scan_x, mu, np.linalg.inv(cov))
                p_array.append(scan_p)
                m_dist_array.append(scan_m_dist)
            else:
                logger.info(f'Cannot find scan {scan_name}')

        return pd.DataFrame(list(zip(file_list, p_array, m_dist_array)), columns=['Scan', 'Probability', 'M-Distance'])


def main():
    """
    Fit a multivariate Gaussian to the data. Get the distribution of positive cases.
    Ouput:
    1. csv file. Probability for each positive sample.
    2. bin file. The pickled data of the Gaussian model.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-res-data-matrix-bin', type=str)
    parser.add_argument('--load-jac-data-matrix-bin', type=str)
    parser.add_argument('--num-res-pc', type=int)
    parser.add_argument('--num-jac-pc', type=int)
    parser.add_argument('--positive-list', type=str)
    parser.add_argument('--out-csv-cancer', type=str)
    parser.add_argument('--out-csv-all', type=str)
    # parser.add_argument('--out-png', type=str)

    args = parser.parse_args()

    fit_obj = FitGaussianJoint()
    fit_obj.load_data(
        args.load_res_data_matrix_bin,
        args.num_res_pc,
        args.load_jac_data_matrix_bin,
        args.num_jac_pc)
    fit_obj.get_distribution(
        read_file_contents_list(args.positive_list),
        args.out_csv_cancer
    )
    fit_obj.get_distribution_all(args.out_csv_all)


if __name__ == '__main__':
    main()