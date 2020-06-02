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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding


logger = get_logger('KMeans')


class ClusterAnalysisDimAnalyzer:
    def __init__(self, data_dict, n_features):
        self._data_dict = data_dict
        self._n_feature = n_features
        self._kmean_n_cluster_range = range(3, 9)
        self._bar_w = 0.12
        self._n_init_kmeans = 10000
        self._con_factor = 0.6

    def plot_kmean_n_cluster_field_list_cancer_subject_first_scan(
            self,
            field_list,
            n_cluster,
            out_png_folder):
        df_field_ori = self._get_dataframe_from_data_dict()
        df_field = df_field_ori[df_field_ori['CancerSubjectFirstScan'] != 0]
        df_field = self._modify_df_field_value(df_field, field_list)

        data_X, _ = self._get_features_and_labels(df_field, 'CancerSubjectFirstScan')

        logger.info(f'Run k-means')
        k_mean = KMeans(n_clusters=n_cluster, n_init=self._n_init_kmeans).fit(data_X)
        pred_labels = k_mean.labels_

        self._plot_bar_figure(pred_labels, field_list, df_field, out_png_folder)

        data_embedded_tsne = self._2D_embed(data_X, pred_labels, 'TSNE')
        self._plot_cluster_2d_embedded(data_embedded_tsne, pred_labels, out_png_folder)

        data_embedded = self._2D_embed(data_X, pred_labels, 'con_TSNE', self._con_factor)
        # self._plot_cluster_2d_embedded(data_embedded, pred_labels, out_png_folder)

        self._plot_cluster_with_label_2d_embedded(data_embedded, field_list, df_field, out_png_folder)

    def _plot_bar_figure(self, pred_labels, field_list, df_field, out_png_folder):
        cluster_size = []
        n_cluster = np.max(pred_labels) + 1
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

            logger.info(f'{field_flag}, AMI = {metric_ami:.2}')
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
        plt.close()

    def _plot_cluster_2d_embedded(self, data_embedded, pred_labels, out_png_folder):
        title = f'KMean clustering, 2D embedding with t-SNE'
        out_png_path = os.path.join(out_png_folder, f'2d_embed_con_TSNE.png')
        self._plot_2D_embedded_data(data_embedded, pred_labels, title, out_png_path)

    def _plot_cluster_with_label_2d_embedded(self, data_embedded, field_list, df_field, out_png_folder):
        for idx_field in range(len(field_list)):
            fig, ax = plt.subplots(figsize=(20, 14))
            field_flag = field_list[idx_field]
            label_list = self._get_field_label_list(field_flag, df_field)
            logger.info(f'Show 2d embedded result for {field_flag}')

            _, data_Y = self._get_features_and_labels(df_field, field_flag)

            for idx_label in range(len(label_list)):
                label_idx_list = np.where(data_Y == idx_label)
                label_data_embedded = data_embedded[label_idx_list]
                ax.scatter(
                    label_data_embedded[:, 0],
                    label_data_embedded[:, 1],
                    alpha=0.9,
                    label=f'{label_list[idx_label]}'
                )
            plt.title(f'2D embedded plot, {field_flag}')
            plt.legend(loc='best')
            out_png_path = os.path.join(out_png_folder, f'2d_embed_label_{field_flag}')
            logger.info(f'Save png to {out_png_path}')
            plt.savefig(out_png_path)
            plt.close()


    def _contract_cluster_data_to_center(self, data_X, pred_label, contract_ratio):
        updated_data_X = np.zeros(data_X.shape, dtype=float)
        num_cluster = np.max(pred_label) + 1
        for idx_cluster in range(num_cluster):
            cluster_idx_list = np.where(pred_label == idx_cluster)
            data_cluster = data_X[cluster_idx_list]
            cluster_centroid = np.average(data_cluster, axis=0)
            contracted_cluster_data = \
                contract_ratio * (data_cluster - cluster_centroid) + cluster_centroid
            updated_data_X[cluster_idx_list] = contracted_cluster_data

        return updated_data_X

    def _2D_embed(self, data_X, pred_label, method, contract_factor=0.7):
        data_embedded = None
        if method == 'LDA':
            lda_obj = LinearDiscriminantAnalysis(n_components=2)
            lda_obj.fit(data_X, pred_label)
            data_embedded = lda_obj.transform(data_X)
        elif method == 'TSNE':
            data_embedded = TSNE(n_components=2).fit_transform(data_X)
        elif method == 'LLE':
            data_embedded = LocallyLinearEmbedding(n_components=2).fit_transform(data_X)
        elif method == 'con_TSNE':
            data_X = self._contract_cluster_data_to_center(data_X, pred_label, contract_factor)
            data_embedded = TSNE(n_components=2).fit_transform(data_X)

        return data_embedded

    def _plot_2D_embedded_data(self, data_embedded, pred_label, title, out_png_path):
        fig, ax = plt.subplots(figsize=(20, 14))
        num_cluster = np.max(pred_label) + 1
        for idx_cluster in range(num_cluster):
            cluster_idx_list = np.where(pred_label == idx_cluster)
            data_embedded_cluster = data_embedded[cluster_idx_list]
            cluster_size = len(cluster_idx_list[0])
            ax.scatter(
                data_embedded_cluster[:, 0],
                data_embedded_cluster[:, 1],
                alpha=0.9,
                label=f'Cluster {idx_cluster + 1} (size {cluster_size})'
            )
        plt.title(title)
        logger.info(f'Save to {out_png_path}')
        plt.legend(loc='best')
        plt.savefig(out_png_path)
        plt.close()

    def _plot_2D_embedding_LDA(self, data_X, pred_label, out_png_folder):
        data_embedded = self._2D_embed(data_X, pred_label, 'LDA')
        title = 'KMean clustering, 2D embedding with LDA'
        out_png_path = os.path.join(out_png_folder, '2d_embed_LDA.png')
        self._plot_2D_embedded_data(data_embedded, pred_label, title, out_png_path)

    def _plot_2D_embedding_tsne(self, data_X, pred_label, out_png_folder):
        data_embedded = self._2D_embed(data_X, pred_label, 'TSNE')
        title = 'KMean clustering, 2D embedding with t-SNE'
        out_png_path = os.path.join(out_png_folder, '2d_embed_tSNE.png')
        self._plot_2D_embedded_data(data_embedded, pred_label, title, out_png_path)

    def _plot_2D_embedding_lle(self, data_X, pred_label, out_png_folder):
        data_embedded = self._2D_embed(data_X, pred_label, 'LLE')
        title = 'KMean clustering, 2D embedding with LLE'
        out_png_path = os.path.join(out_png_folder, '2d_embed_LLE.png')
        self._plot_2D_embedded_data(data_embedded, pred_label, title, out_png_path)

    def _plot_2D_embedding_con_TSNE_series(self, data_X, pred_label, out_png_folder):
        factor_idx = 0
        for con_factor in np.arange(0.5, 1.0, 0.05):
            data_embedded = self._2D_embed(data_X, pred_label, f'con_TSNE', con_factor)
            title = f'KMean clustering, 2D embedding with con_TSNE, contract factor {con_factor}'
            out_png_path = os.path.join(out_png_folder, f'2d_embed_con_TSNE_{factor_idx}.png')
            self._plot_2D_embedded_data(data_embedded, pred_label, title, out_png_path)
            factor_idx += 1

    def _plot_2D_embedding_con_TSNE(self, data_X, pred_label, out_png_folder):
        con_factor = 0.7
        data_embedded = self._2D_embed(data_X, pred_label, f'con_TSNE', con_factor)
        title = f'KMean clustering, 2D embedding with t-SNE'
        out_png_path = os.path.join(out_png_folder, f'2d_embed_con_TSNE.png')
        self._plot_2D_embedded_data(data_embedded, pred_label, title, out_png_path)

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
        elif field_flag == 'CancerSubjectFirstScan':
            df_field = ori_df[ori_df[field_flag] != 0]
            df_field = df_field.fillna(value={field_flag: 0})
            print(f'df, Cancer first scan count {df_field[df_field[field_flag] == 1].shape[0]}')
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
        elif field_flag == 'CancerSubjectFirstScan':
            label_list.append(f'non-cancer ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'cancer, first scan ({self._count_num_field(df_field, field_flag, 1)})')
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
            # if ('CancerSubjectFirstScan' in new_data_item):
            #     count_first_cancer += 1
            if "Cancer" in new_data_item:
                if new_data_item['Cancer'] == 1:
                    count_cancer += 1
            # if ("Cancer" in new_data_item):
            #     count_cancer += 1
        logger.info(f'Count first cancer: {count_first_cancer}')
        logger.info(f'Count cancer: {count_cancer}')
        df = pd.DataFrame.from_dict(new_data_dict, orient='index')
        return df

    @staticmethod
    def _count_num_field(df, field_flag, field_val):
        return df[df[field_flag] == field_val].shape[0]

    @staticmethod
    def _get_pc_str(idx):
        return f'pc{idx}'

def main():
    parser = argparse.ArgumentParser(description='KMean clustering analysis')
    parser.add_argument('--in-data-dict-bin', type=str)
    parser.add_argument('--n-features', type=int)
    parser.add_argument('--out-png-folder', type=str)
    parser.add_argument('--n-cluster', type=int, default=10)
    args = parser.parse_args()

    in_data_dict = load_object(args.in_data_dict_bin)
    kmean_analyzer = ClusterAnalysisDimAnalyzer(in_data_dict, args.n_features)

    kmean_analyzer.plot_kmean_n_cluster_field_list_cancer_subject_first_scan(
        ['CancerSubjectFirstScan', 'COPD', 'Coronary Artery Calcification', 'Age', 'Packyear', 'bmi'],
        args.n_cluster, args.out_png_folder)


if __name__ == '__main__':
    main()