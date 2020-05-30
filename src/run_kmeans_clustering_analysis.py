import argparse
from tools.clinical import ClinicalDataReaderSPORE
from tools.data_io import load_object
from tools.utils import get_logger
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score


logger = get_logger('KMeans')


class KMeansClusterAnalyzer:
    def __init__(self, label_df):
        self._label_df = label_df
        self._n_feature = 20
        self._kmean_n_cluster_range = range(3, 9)
        self._bar_w = 0.12
        self._n_init = 1000

    def plot_kmean_n_cluster_field_list_exclude_cancer_in_1_year(self, field_list, n_cluster, out_png_folder):
        df_field = self._label_df[self._label_df['CancerIncubation'] != 0]
        df_field = self._modify_df_field_value(df_field, field_list)

        data_X, _ = self._get_features_and_labels(df_field, 'CancerIncubation')

        logger.info(f'Run k-means')
        k_mean = KMeans(n_clusters=n_cluster, n_init=self._n_init).fit(data_X)
        pred_labels = k_mean.labels_

        cluster_size = []
        for idx_cluster in range(n_cluster):
            cluster_idx_list = np.where(pred_labels == idx_cluster)
            cluster_size.append(len(cluster_idx_list[0]))

        # Plot
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 3)

        for idx_field in range(len(field_list)):
            ax = plt.subplot(gs[idx_field])

            field_flag = field_list[idx_field]
            label_list = self._get_field_label_list(field_flag, df_field)
            logger.info(f'Label result for {field_flag}')

            _, data_Y = self._get_features_and_labels(df_field, field_flag)

            true_label_ratio_per_cluster = np.zeros((len(label_list), n_cluster), dtype=float)
            for idx_cluster in range(n_cluster):
                cluster_idx_list = np.where(pred_labels == idx_cluster)
                cluster_true_label_list = data_Y[cluster_idx_list]
                for idx_label in range(len(label_list)):
                    true_label_ratio_per_cluster[idx_label, idx_cluster] = \
                        np.count_nonzero(cluster_true_label_list == idx_label) / \
                        len(cluster_idx_list[0])

            base_x_locs = np.arange(n_cluster)
            for idx_label in range(len(label_list)):
                off_set = self._bar_w * (idx_label - 0.5 * len(label_list) + 0.5)
                ax.bar(base_x_locs + off_set, true_label_ratio_per_cluster[idx_label, :],
                       width=self._bar_w, align='center', label=label_list[idx_label])

            metric_ami = adjusted_mutual_info_score(data_Y, pred_labels)

            plt.title(f'{field_flag}, KMean {n_cluster} cluster, AMI {metric_ami:.2}')
            plt.legend(loc='best')

            plt.xlim(left=-1, right=n_cluster)
            x_tick_pos = np.arange(n_cluster)
            ax.set_xticks(x_tick_pos)
            ax.set_xticklabels(cluster_size)

            plt.xlabel('Cluster size')
            plt.ylabel('Label ratio')

        out_png_path = os.path.join(out_png_folder, f'exclude_cancer_in_1_year.png')
        logger.info(f'Save to {out_png_path}')
        plt.savefig(out_png_path)

    def plot_cancer_incubation_distribute(self, out_png_folder):
        # TODO
        pass

    def plot_kmean_series(self, field_flag, out_png_folder):
        df_field, label_list = self._get_df_field(field_flag)
        data_X, data_Y = self._get_features_and_labels(df_field, field_flag)

        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(2, 3)
        idx_n_cluster = 0
        for n_cluster in self._kmean_n_cluster_range:
            ax = plt.subplot(gs[idx_n_cluster])

            logger.info(f'Run k-means with {n_cluster} clusters')
            k_mean = KMeans(n_clusters=n_cluster, n_init=self._n_init).fit(data_X)
            pred_labels = k_mean.labels_

            # cluster_ratio = []
            cluster_size = []
            true_label_ratio_per_cluster = np.zeros((len(label_list), n_cluster), dtype=float)
            for idx_cluster in range(n_cluster):
                cluster_idx_list = np.where(pred_labels == idx_cluster)
                cluster_true_label_list = data_Y[cluster_idx_list]
                # cluster_ratio.append(len(cluster_idx_list[0]) / len(pred_labels))
                cluster_size.append(len(cluster_idx_list[0]))
                for idx_label in range(len(label_list)):
                    true_label_ratio_per_cluster[idx_label, idx_cluster] = \
                        np.count_nonzero(cluster_true_label_list == idx_label) / \
                        len(cluster_idx_list[0])

            base_x_locs = np.arange(n_cluster)
            for idx_label in range(len(label_list)):
                off_set = self._bar_w * (idx_label - 0.5 * len(label_list) + 0.5)
                ax.bar(base_x_locs + off_set, true_label_ratio_per_cluster[idx_label, :],
                       width=self._bar_w, align='center', label=label_list[idx_label])

            metric_ami = adjusted_mutual_info_score(data_Y, pred_labels)

            plt.title(f'KMean - {n_cluster} cluster, AMI {metric_ami:.2}')
            plt.legend(loc='best')

            plt.xlim(left=-1, right=n_cluster)
            x_tick_pos = np.arange(n_cluster)
            ax.set_xticks(x_tick_pos)
            # ax.set_xticklabels(f'{cluster_ratio:.2}' for cluster_ratio in cluster_ratio)
            ax.set_xticklabels(cluster_size)

            plt.xlabel('Cluster size')
            plt.ylabel('Label ratio')

            idx_n_cluster += 1

        out_png_path = os.path.join(out_png_folder, f'kmean_series_{field_flag}.png')
        logger.info(f'Save kmean_series to {out_png_path}')
        plt.savefig(out_png_path)

    def _get_features_and_labels(self, df_field, field_flag):
        n_sample = df_field.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)
        for feature_idx in range(self._n_feature):
            pc_str = self._get_pc_str(feature_idx)
            data_X[:, feature_idx] = df_field[pc_str].tolist()[:]
        data_Y = df_field[field_flag].tolist()
        data_Y = np.array(data_Y)

        return data_X, data_Y

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
        else:
            raise NotImplementedError

        return df_field

    def _modify_df_field_value(self, ori_df, field_flag_list):
        for field_flag in field_flag_list:
            ori_df = self._modify_df_one_field(ori_df, field_flag)

        return ori_df

    def _get_field_label_list(self, field_flag, df_field):
        label_list = []
        if (field_flag == 'copd') | (field_flag == 'COPD'):
            label_list.append(f'copd:no ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'copd:yes ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'copd:unknown ({self._count_num_field(df_field, field_flag, 2)})')
        elif field_flag == 'Age':
            label_list.append(f'Age<60 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'60<=Age<70 ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'Age>=70 ({self._count_num_field(df_field, field_flag, 2)})')
        elif (field_flag == 'packyearsreported') | (field_flag == 'Packyear'):
            label_list.append(f'packyear<35 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'35<=packyear<60 ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'packyear>=60 ({self._count_num_field(df_field, field_flag, 2)})')
        elif (field_flag == 'Coronary Artery Calcification') | (field_flag == 'CAC'):
            label_list.append(f'CAC: None ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'CAC: Mild ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'CAC: Moderate ({self._count_num_field(df_field, field_flag, 2)})')
            label_list.append(f'CAC: Severe ({self._count_num_field(df_field, field_flag, 3)})')
        elif field_flag == 'bmi':
            label_list.append(f'BMI < 21 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'21 <= BMI < 35 ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'BMI >= 35 ({self._count_num_field(df_field, field_flag, 2)})')
        elif (field_flag == 'cancer_bengin') | (field_flag == 'Cancer'):
            label_list.append(f'non-cancer ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'cancer ({self._count_num_field(df_field, field_flag, 1)})')
        elif field_flag == 'CancerIncubation':
            label_list.append(f'non-cancer ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'cancer, time to diag >= 1y ({self._count_num_field(df_field, field_flag, 1)})')
        else:
            raise NotImplementedError

        return label_list

    def _get_df_field(self, field_flag):
        df_field = None
        label_list = []
        if (field_flag == 'copd') | (field_flag == 'COPD'):
            df_field = self._label_df.fillna(value={field_flag: 2})
            df_field = df_field.replace({field_flag: {'Yes': 1, 'No': 0}})
            label_list.append(f'copd:no')
            label_list.append(f'copd:yes')
            label_list.append(f'copd:unknown')
        elif field_flag == 'Age':
            df_field = self._label_df
            df_field.loc[self._label_df[field_flag] < 60, field_flag] = 0
            df_field.loc[(self._label_df[field_flag] < 70) & (self._label_df[field_flag] >= 60), field_flag] = 1
            df_field.loc[self._label_df[field_flag] >= 70, field_flag] = 2
            label_list.append(f'Age<60 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'60<=Age<70 ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'Age>=70 ({self._count_num_field(df_field, field_flag, 2)})')
        elif (field_flag == 'packyearsreported') | (field_flag == 'Packyear'):
            df_field = self._label_df

            df_field.loc[self._label_df[field_flag] < 35, field_flag] = 0
            df_field.loc[(self._label_df[field_flag] >= 35) & (self._label_df[field_flag] < 60), field_flag] = 1
            df_field.loc[self._label_df[field_flag] >= 60, field_flag] = 2

            label_list.append(f'packyear<35 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'35<=packyear<60 ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'packyear>=60 ({self._count_num_field(df_field, field_flag, 2)})')
        elif (field_flag == 'Coronary Artery Calcification') | (field_flag == 'CAC'):
            df_field = self._label_df[
                (self._label_df[field_flag] == 'Severe') |
                (self._label_df[field_flag] == 'Moderate') |
                (self._label_df[field_flag] == 'Mild') |
                (self._label_df[field_flag] == 'None')
                ]
            df_field.loc[df_field[field_flag] == 'None', field_flag] = 0
            df_field.loc[df_field[field_flag] == 'Mild', field_flag] = 1
            df_field.loc[df_field[field_flag] == 'Moderate', field_flag] = 2
            df_field.loc[df_field[field_flag] == 'Severe', field_flag] = 3

            df_field = df_field.replace(
                {'Coronary Artery Calcification':
                     {'Severe': 3, 'Moderate': 2, 'Mild': 1, 'None': 0}}
            )

            label_list.append(f'CAC: None ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'CAC: Mild ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'CAC: Moderate ({self._count_num_field(df_field, field_flag, 2)})')
            label_list.append(f'CAC: Severe ({self._count_num_field(df_field, field_flag, 3)})')
        elif field_flag == 'bmi':
            df_field = self._label_df
            df_field.loc[self._label_df[field_flag] < 21, field_flag] = 0
            df_field.loc[(self._label_df[field_flag] >= 21) & (self._label_df[field_flag] < 35), field_flag] = 1
            df_field.loc[self._label_df[field_flag] >= 35, field_flag] = 2
            label_list.append(f'BMI < 21 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'21 <= BMI < 35 ({self._count_num_field(df_field, field_flag, 1)})')
            label_list.append(f'BMI >= 35 ({self._count_num_field(df_field, field_flag, 2)})')
        elif (field_flag == 'cancer_bengin') | (field_flag == 'Cancer'):
            df_field = self._label_df

            label_list.append(f'non-cancer ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'cancer ({self._count_num_field(df_field, field_flag, 1)})')
        elif field_flag == 'CancerIncubation':
            df_field = self._label_df[self._label_df[field_flag] != 0]
            df_field = df_field.fillna(value={field_flag: 0})
            label_list.append(f'non-cancer ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'cancer, time to diag >= 1y ({self._count_num_field(df_field, field_flag, 1)})')
        else:
            raise NotImplementedError

        return df_field, label_list

    @staticmethod
    def _count_num_field(df, field_flag, field_val):
        return df[df[field_flag] == field_val].shape[0]

    @staticmethod
    def _get_pc_str(idx):
        return f'pc{idx}'

def main():
    parser = argparse.ArgumentParser(description='KMean clustering analysis')
    parser.add_argument('--data-csv', type=str, )
    parser.add_argument('--out-png-folder', type=str)
    args = parser.parse_args()

    data_df = pd.read_csv(args.data_csv)
    kmean_analyzer = KMeansClusterAnalyzer(data_df)

    # kmean_analyzer.plot_kmean_series('COPD', args.out_png_folder)
    # kmean_analyzer.plot_kmean_series('bmi', args.out_png_folder)
    # kmean_analyzer.plot_kmean_series('Cancer', args.out_png_folder)
    # kmean_analyzer.plot_kmean_series('CAC', args.out_png_folder)
    # kmean_analyzer.plot_kmean_series('Age', args.out_png_folder)
    # kmean_analyzer.plot_kmean_series('Packyear', args.out_png_folder)
    # kmean_analyzer.plot_kmean_series('CancerIncubation', args.out_png_folder)

    kmean_analyzer.plot_kmean_n_cluster_field_list_exclude_cancer_in_1_year(
        ['CancerIncubation', 'COPD', 'Coronary Artery Calcification', 'Age', 'Packyear', 'bmi'],
        9, args.out_png_folder)


if __name__ == '__main__':
    main()