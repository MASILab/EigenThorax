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


class FitGaussian:
    def __init__(self):
        self._in_data_matrix_obj = None
        self._num_pc = None
        self._file_list = None
        self._use_data_matrix = None
        self._gaussian_model = None

    def load_data(self, in_data_matrix_bin_path, num_pc):
        logger.info(f'Load bin data file {in_data_matrix_bin_path}')
        self._data_obj = load_object(in_data_matrix_bin_path)

        self._num_pc = num_pc
        self._file_list = self._data_obj['file_list']

        self._use_data_matrix = self._data_obj['projected_matrix'][:, :self._num_pc]

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

    def plot_2d_distribution(self, out_png):
        """
        A testing function for simplified 2d case
        :param out_png:
        :return:
        """
        if self._num_pc != 2:
            logger.info(f'2D plot require the number of pc equals to 2, current dimension is {self._num_pc}')
            return

        # 1. Get range
        x_min = np.min(self._use_data_matrix[:, 0])
        x_max = np.max(self._use_data_matrix[:, 0])
        x_min, x_max = x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min)
        y_min = np.min(self._use_data_matrix[:, 1])
        y_max = np.max(self._use_data_matrix[:, 1])
        y_min, y_max = y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)
        x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        pos = np.dstack((x, y))

        # 2. plot the distribution
        fig, ax = plt.subplots()
        im_contour = ax.contourf(x, y, self._gaussian_model.pdf(pos))

        # 3. scatter plot the samples
        ax.scatter(
            self._use_data_matrix[:, 0],
            self._use_data_matrix[:, 1],
            marker='x',
            c='r',
            s=0.5
        )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cb = plt.colorbar(im_contour, cax=cax)
        cb.set_label('PDF')

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0)


def main():
    """
    Fit a multivariate Gaussian to the data. Get the distribution of positive cases.
    Ouput:
    1. csv file. Probability for each positive sample.
    2. bin file. The pickled data of the Gaussian model.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-data-matrix-bin', type=str)
    parser.add_argument('--positive-list', type=str)
    parser.add_argument('--out-csv-cancer', type=str)
    parser.add_argument('--out-csv-all', type=str)
    parser.add_argument('--out-png', type=str)
    parser.add_argument('--num-pc', type=int)
    args = parser.parse_args()

    fit_obj = FitGaussian()
    fit_obj.load_data(args.load_data_matrix_bin, args.num_pc)
    fit_obj.get_distribution(
        read_file_contents_list(args.positive_list),
        args.out_csv_cancer
    )
    fit_obj.get_distribution_all(args.out_csv_all)
    fit_obj.plot_2d_distribution(args.out_png)


if __name__ == '__main__':
    main()