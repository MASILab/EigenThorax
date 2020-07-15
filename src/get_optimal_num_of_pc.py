import argparse
from tools.pca import PCA_NII_3D
from tools.data_io import ScanWrapper, DataFolder, save_object, load_object
from tools.utils import get_logger
from tools.clinical import ClinicalDataReaderSPORE
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


logger = get_logger('Optimal # of PC')


class OptimizeNumberOfPC:
    def __init__(self, data_bin_path):
        self._data_bin_path = data_bin_path
        self._data_obj = None

    def load_data(self):
        self._data_obj = load_object(self._data_bin_path)

    def plot_ratio_against_first_n_pc(self, max_n_pc, out_png_folder):
        ratio_array_euclidean = np.zeros((max_n_pc,), dtype=float)
        ratio_array_cosine = np.zeros((max_n_pc,), dtype=float)

        intra_pair_list, inter_pair_list = self._get_idx_pair_of_intra_inter_groups()

        for idx_n_pc in range(max_n_pc):
            logger.info(f'Get the ratio when using {idx_n_pc} components')
            n_pc_used = idx_n_pc + 1
            ratio_array_euclidean[idx_n_pc] = self._get_intra_inter_ratio(
                intra_pair_list,
                inter_pair_list,
                n_pc_used,
                'euclidean'
            )
            ratio_array_cosine[idx_n_pc] = self._get_intra_inter_ratio(
                intra_pair_list,
                inter_pair_list,
                n_pc_used,
                'cosine'
            )

        logger.info(f'The average distance ratio')
        logger.info(f'Euclidean')
        print(ratio_array_euclidean)
        logger.info(f'Cosine')
        print(ratio_array_cosine)

        plt.figure(figsize=(30, 15))

        font = {'weight': 'bold',
                'size': 22}

        matplotlib.rc('font', **font)

        x_list = range(1, max_n_pc + 1)
        ax = plt.subplot()

        ax.plot(x_list,
                ratio_array_euclidean,
                marker='+',
                linewidth=2,
                label='Euclidean')

        ax.plot(x_list,
                ratio_array_cosine,
                marker='+',
                linewidth=2,
                label='Cosine')

        ax.legend(loc='best')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Averaged Intra-subject Distance / Averaged Inter-subject Distance')

        out_png_path = os.path.join(out_png_folder, 'average_dist_ratio.png')
        plt.savefig(out_png_path, bbox_inches='tight', pad_inches=1)

    def plot_ratio_against_each_pc(self, max_n_pc, out_png_folder):
        ratio_array_euclidean = np.zeros((max_n_pc,), dtype=float)

        intra_pair_list, inter_pair_list = self._get_idx_pair_of_intra_inter_groups()

        for idx_n_pc in range(max_n_pc):
            logger.info(f'Get ratio for pc {idx_n_pc}')
            ratio_array_euclidean[idx_n_pc] = self._get_intra_inter_ratio_per_pc(
                intra_pair_list,
                inter_pair_list,
                idx_n_pc
            )

        logger.info(f'The average distance ratio')
        logger.info(f'Euclidean')
        print(ratio_array_euclidean)

        plt.figure(figsize=(30, 15))

        font = {'weight': 'bold',
                'size': 22}

        matplotlib.rc('font', **font)

        x_list = range(1, max_n_pc + 1)
        ax = plt.subplot()

        ax.plot(x_list,
                ratio_array_euclidean,
                marker='+',
                linewidth=2,
                label='Euclidean')

        ax.legend(loc='best')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Averaged Intra-subject Distance / Averaged Inter-subject Distance')

        out_png_path = os.path.join(out_png_folder, 'average_dist_ratio_per_pc.png')
        plt.savefig(out_png_path, bbox_inches='tight', pad_inches=1)

    def _get_intra_inter_ratio_per_pc(
            self,
            intra_pair_list,
            inter_pair_list,
            idx_pc_used
    ):
        intra_average = self._get_average_dist_pair_list_per_pc(
            intra_pair_list,
            idx_pc_used
        )
        inter_average = self._get_average_dist_pair_list_per_pc(
            inter_pair_list,
            idx_pc_used
        )

        return intra_average / inter_average

    def _get_average_dist_pair_list_per_pc(
            self,
            pair_list,
            pc_idx
    ):
        data_vec = np.zeros((self._data_obj['projected_matrix'].shape[0], 1),
                            dtype=float)
        data_vec[:, 0] = self._data_obj['projected_matrix'][:, pc_idx]

        distance_matrix = pairwise_distances(X=data_vec)

        pair_distance_array = np.array(
            [distance_matrix[pair_item[0], pair_list[1]]
             for pair_item in pair_list])

        return np.mean(pair_distance_array)

    def _get_intra_inter_ratio(self,
                               intra_pair_list,
                               inter_pair_list,
                               n_pc_used,
                               metric_flag):
        # intra_pair_list, inter_pair_list = self._get_idx_pair_of_intra_inter_groups()

        intra_average = self._get_average_dist_first_n_pc(intra_pair_list, n_pc_used, metric_flag)
        inter_average = self._get_average_dist_first_n_pc(inter_pair_list, n_pc_used, metric_flag)

        return intra_average / inter_average

    def _get_idx_pair_of_intra_inter_groups(self):
        file_list = self._data_obj['file_list']

        intra_pair_list = []
        inter_pair_list = []

        for idx1 in range(len(file_list)):
            file_name1 = file_list[idx1]
            for idx2 in range(idx1 + 1, len(file_list)):
                file_name2 = file_list[idx2]
                idx_pair = [idx1, idx2]
                if self._check_if_same_subject(file_name1, file_name2):
                    intra_pair_list.append(idx_pair)
                else:
                    inter_pair_list.append(idx_pair)

        logger.info(f'Num of intra-subject pair: {len(intra_pair_list)}')
        logger.info(f'Num of inter-subject pair: {len(inter_pair_list)}')

        logger.info('List of intra subject pairs')
        print(intra_pair_list)

        return intra_pair_list, inter_pair_list

    def _check_if_same_subject(self, file_name1, file_name2):
        subject1_id = ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name1)
        subject2_id = ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name2)

        return subject1_id == subject2_id

    def _get_average_euclidean_dist_first_n_pc(self, pair_list, n_pc_used):
        return self._get_average_dist_first_n_pc(pair_list, n_pc_used, 'euclidean')

    def _get_average_cosine_dist_first_n_pc(self, pair_list, n_pc_used):
        return self._get_average_dist_first_n_pc(pair_list, n_pc_used, 'cosine')

    def _get_average_dist_first_n_pc(self, pair_list, n_pc_used, metric_type):
        data_matrix_fist_n = self._data_obj['projected_matrix'][:, :n_pc_used]

        distance_matrix = pairwise_distances(
            X=data_matrix_fist_n,
            metric=metric_type
        )

        pair_distance_array = np.array(
            [distance_matrix[pair_item[0], pair_list[1]]
             for pair_item in pair_list])

        return np.mean(pair_distance_array)


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-data-matrix-bin', type=str)
    parser.add_argument('--out-png-folder', type=str)
    parser.add_argument('--num-pc', type=int)
    args = parser.parse_args()

    dr_obj = OptimizeNumberOfPC(
        args.load_data_matrix_bin
    )

    dr_obj.load_data()
    dr_obj.plot_ratio_against_first_n_pc(args.num_pc, args.out_png_folder)
    dr_obj.plot_ratio_against_each_pc(args.num_pc, args.out_png_folder)


if __name__ == '__main__':
    main()