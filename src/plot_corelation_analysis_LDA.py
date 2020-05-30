import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import MaxNLocator
from tools.clinical import ClinicalDataReaderSPORE
from tools.data_io import load_object
from tools.utils import get_logger
import pandas as pd
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.gridspec as gridspec
import matplotlib.mlab as mlab
import datetime


logger = get_logger('LDA')


class PlotCorrAnalyzeLDA:
    def __init__(self, label_df):
        self._label_df = label_df
        self._n_feature = 20
        self._metric_dict = {}
        self._roc_data = {}

    def save_label_file(self, out_csv):
        self._label_df.to_csv(out_csv)

    def _train_LDA(self, target_df, field_flag, n_components):
        n_sample = target_df.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)

        for feature_idx in range(self._n_feature):
            pc_str = self.get_pc_str(feature_idx)
            data_X[:, feature_idx] = target_df[pc_str].tolist()[:]

        data_Y = target_df[field_flag].tolist()

        # LDA
        lda_obj = LinearDiscriminantAnalysis(n_components=n_components)
        lda_obj.fit(data_X, data_Y)

        return lda_obj

    def _plot_LDA_proj(self, target_df, field_flag, out_png):
        lda_obj = self._train_LDA(target_df, field_flag, n_components=1)

        # Project unit vector of each feature dimension.
        zero_vector = np.zeros((1, self._n_feature), dtype=float)
        unit_vectors = np.zeros((self._n_feature, self._n_feature), dtype=float)
        np.fill_diagonal(unit_vectors, 1)
        projected_unit = lda_obj.transform(unit_vectors)
        projected_zero = lda_obj.transform(zero_vector)
        projected_unit = projected_unit - projected_zero

        # Plot
        plt.figure(figsize=(16, 10))
        plt.bar(range(self._n_feature), projected_unit[:, 0])
        plt.title(field_flag)

        logger.info(f'Save to {out_png}')
        plt.savefig(out_png)
        plt.close()

    def _plot_LDA_subspace_2d(self, target_df, field_flag, out_png):
        lda_obj = self._train_LDA(target_df, field_flag, n_components=2)

        n_sample = target_df.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)

        for feature_idx in range(self._n_feature):
            pc_str = self.get_pc_str(feature_idx)
            data_X[:, feature_idx] = target_df[pc_str].tolist()[:]

        data_Y = target_df[field_flag].tolist()
        projected_X = lda_obj.transform(data_X)

        logger.info(projected_X.shape)

        # Plot
        plt.figure(figsize=(16, 10))
        plt.scatter(
            projected_X[:, 0],
            projected_X[:, 1],
            c=data_Y,
            cmap='bwr',
            alpha=0.3
        )

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png)

    def _plot_LDA_subspace_1d(self, target_df, label_list, field_flag, out_png):
        lda_obj = self._train_LDA(target_df, field_flag, n_components=1)
        n_sample = target_df.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)

        for feature_idx in range(self._n_feature):
            pc_str = self.get_pc_str(feature_idx)
            data_X[:, feature_idx] = target_df[pc_str].tolist()[:]

        data_Y = target_df[field_flag].tolist()
        projected_X = lda_obj.transform(data_X)

        logger.info(projected_X.shape)

        # Plot
        y_vals = np.random.normal(0, 0.01, projected_X.shape[0])
        fig, ax = plt.subplots(figsize=(16, 10))

        plt.scatter(
            projected_X[:, 0],
            y_vals + data_Y,
            c=data_Y,
            cmap='jet',
            alpha=0.3
        )

        plt.ylim(bottom=np.min(data_Y)-1, top=np.max(data_Y)+1)

        # label
        y_tick_pos = np.arange(len(label_list))
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(label_list)
        ax.tick_params(which='minor', labelsize=20)

        plt.title(field_flag)

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png)

        # Log the discriminant metric
        metric = self._get_discriminant_metric(projected_X[:, 0], data_Y)
        self._metric_dict[field_flag] = {
            'Metric': metric
        }

    def _get_5_fold_roc_input(self, target_df, field_flag):
        n_sample = target_df.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)
        for feature_idx in range(self._n_feature):
            pc_str = self.get_pc_str(feature_idx)
            data_X[:, feature_idx] = target_df[pc_str].tolist()[:]
        data_Y = target_df[field_flag].tolist()
        data_Y = np.array(data_Y)

        # convert multi-class to single. only consider the class with largest label num.
        num_classes = np.max(data_Y) + 1

        estimate_prob = np.zeros((n_sample,), dtype=float)
        n_fold = KFold(n_splits=5)
        logger.info('Run 5 fold LDA')
        for train_idx, test_idx in n_fold.split(data_X):
            data_X_train, data_X_test = data_X[train_idx], data_X[test_idx]
            data_Y_train, data_Y_test = data_Y[train_idx], data_Y[test_idx]

            lda_obj = LinearDiscriminantAnalysis(n_components=1)
            lda_obj.fit(data_X_train, data_Y_train)

            # print(data_X_test)
            # print(num_classes)
            # print(test_idx)
            estimate_prob[test_idx] = lda_obj.predict_proba(data_X_test)[:, int(num_classes) - 1]

            logger.info(f'Num of positive sample in test group {np.sum(data_Y_test)}')

        return estimate_prob, (data_Y == num_classes - 1).astype(int)

    def _get_5_fold_roc_separate_input(self, target_df, field_flag):
        n_sample = target_df.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)
        for feature_idx in range(self._n_feature):
            pc_str = self.get_pc_str(feature_idx)
            data_X[:, feature_idx] = target_df[pc_str].tolist()[:]
        data_Y = target_df[field_flag].tolist()
        data_Y = np.array(data_Y)

        # convert multi-class to single. only consider the class with largest label num.
        num_classes = np.max(data_Y) + 1

        # estimate_prob = np.zeros((n_sample,), dtype=float)
        estimate_prob_list = []
        data_Y_list = []
        projected_list = []
        n_fold = KFold(n_splits=5)

        # data_Y = np.array(data_Y, dtype=int)
        negative_index_list = np.where(data_Y == 0)[0]
        positive_index_list = np.where(data_Y == 1)[0]

        train_idx_list = []
        test_idx_list = []

        for neg_train_idx_of_idx, neg_test_idx_of_idx in n_fold.split(negative_index_list):
            neg_train_idx, neg_test_idx = \
                negative_index_list[neg_train_idx_of_idx], negative_index_list[neg_test_idx_of_idx]
            train_idx_list.append(neg_train_idx)
            test_idx_list.append(neg_test_idx)

        idx_fold = 0
        for pos_train_idx_of_idx, pos_test_idx_of_idx in n_fold.split(positive_index_list):
            pos_train_idx, pos_test_idx = \
                positive_index_list[pos_train_idx_of_idx], positive_index_list[pos_test_idx_of_idx]
            train_idx_list[idx_fold] = np.concatenate((train_idx_list[idx_fold], pos_train_idx))
            test_idx_list[idx_fold] = np.concatenate((test_idx_list[idx_fold], pos_test_idx))
            idx_fold += 1

        logger.info('Run 5 fold LDA')
        for idx_fold in range(5):
            train_idx = train_idx_list[idx_fold]
            test_idx = test_idx_list[idx_fold]

            data_X_train, data_X_test = data_X[train_idx], data_X[test_idx]
            data_Y_train, data_Y_test = data_Y[train_idx], data_Y[test_idx]

            lda_obj = LinearDiscriminantAnalysis(n_components=1)
            lda_obj.fit(data_X_train, data_Y_train)

            estimate_prob = lda_obj.predict_proba(data_X_test)[:, int(num_classes) - 1]
            estimate_prob_list.append(estimate_prob)
            data_Y_list.append((data_Y_test == num_classes - 1).astype(int))
            projected = lda_obj.transform(data_X_test)
            projected_list.append(projected[:, 0])
            logger.info(f'Num of positive sample in test group {np.sum(data_Y_test)}')

        return estimate_prob_list, data_Y_list, projected_list

    def _get_roc_input(self, target_df, field_flag):
        lda_trained = self._train_LDA(target_df, field_flag, n_components=1)

        n_sample = target_df.shape[0]
        data_X = np.zeros((n_sample, self._n_feature), dtype=float)
        for feature_idx in range(self._n_feature):
            pc_str = self.get_pc_str(feature_idx)
            data_X[:, feature_idx] = target_df[pc_str].tolist()[:]
        data_Y = target_df[field_flag].tolist()

        estimate_prob = lda_trained.predict_proba(data_X)

        # print(estimate_prob)
        return estimate_prob[:, 1], data_Y

    def save_metric(self, out_csv):
        metric_df = pd.DataFrame.from_dict(self._metric_dict, orient='index')
        print(metric_df)
        logger.info(f'Save metric csv to {out_csv}')
        metric_df.to_csv(out_csv)

    def save_5_fold_roc_separate(self, field_list, out_folder):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3)

        auc_dict = {}

        idx_field = 0
        for field_flag in field_list:
            logger.info(f'Run 5 fold LDA on {field_flag}')
            df_field, label_list = self._get_df_field(field_flag)
            prob_list, gt_label_list = self._get_5_fold_roc_separate_input(df_field, field_flag)

            field_auc_dict = {}
            ax = plt.subplot(gs[idx_field])
            for idx_fold in range(5):
                prob = prob_list[idx_fold]
                gt_label = gt_label_list[idx_fold]
                auc_val = round(metrics.roc_auc_score(gt_label, prob), 3)
                fpr, tpr, _ = metrics.roc_curve(gt_label, prob, pos_label=1)
                # precision, recall, _ = metrics.precision_recall_curve(gt_label, prob)
                # print(precision)
                # print(recall)
                # auc_val = round(metrics.auc(recall, precision), 3)
                num_pos_test = np.sum(gt_label)
                plt.plot(fpr, tpr, label=f'Fold: {idx_fold+1}, # pos val: {num_pos_test}, AUC: {auc_val}')
                # plt.plot(recall, precision, label=f'Fold: {idx_fold + 1}, # pos val: {num_pos_test}, AUC (PR): {auc_val}')
                field_auc_dict[f'fold_{idx_fold+1}'] = auc_val

            # plt.xlabel('False positive rate')
            # plt.ylabel('True positive rate')
            plt.title(f'ROC - {field_flag}')
            plt.legend(loc='best')

            auc_dict[field_flag] = field_auc_dict

            idx_field += 1

        roc_png_path = os.path.join(out_folder, 'roc_separate.png')
        logger.info(f'Save roc png to {roc_png_path}')
        plt.savefig(roc_png_path)

        # auc_csv_path = os.path.join(out_folder, 'auc_separate.csv')
        # df_auc = pd.DataFrame.from_dict(auc_dict, orient='index')
        # logger.info(f'Save auc csv to {auc_csv_path}')
        # df_auc.to_csv(auc_csv_path)

    def save_5_fold_roc_pr_separate(self, field_list, out_folder):
        fig = plt.figure(figsize=(16, 10))
        # gs = gridspec.GridSpec(2, 3)
        gs = gridspec.GridSpec(1, 2)

        idx_field = 0
        for field_flag in field_list:
            logger.info(f'Run 5 fold LDA on {field_flag}')
            df_field, label_list = self._get_df_field(field_flag)
            prob_list, gt_label_list = self._get_5_fold_roc_separate_input(df_field, field_flag)

            ax = plt.subplot(gs[0])
            for idx_fold in range(5):
                prob = prob_list[idx_fold]
                gt_label = gt_label_list[idx_fold]
                auc_val = round(metrics.roc_auc_score(gt_label, prob), 3)
                fpr, tpr, _ = metrics.roc_curve(gt_label, prob, pos_label=1)
                num_pos_test = np.sum(gt_label)
                plt.plot(fpr, tpr, label=f'Fold: {idx_fold+1}, # pos val: {num_pos_test}, AUC: {auc_val}')

            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title(f'ROC - {field_flag}')
            plt.legend(loc='best')

            ax = plt.subplot(gs[1])
            for idx_fold in range(5):
                prob = prob_list[idx_fold]
                gt_label = gt_label_list[idx_fold]
                precision, recall, _ = metrics.precision_recall_curve(gt_label, prob)
                auc_val = round(metrics.auc(recall, precision), 3)
                num_pos_test = np.sum(gt_label)
                plt.plot(recall, precision, label=f'Fold: {idx_fold + 1}, # pos val: {num_pos_test}, AUC (PR): {auc_val}')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Recall-precision - {field_flag}')
            plt.legend(loc='best')

            idx_field += 1

        roc_png_path = os.path.join(out_folder, 'roc_separate.png')
        logger.info(f'Save roc png to {roc_png_path}')
        plt.savefig(roc_png_path)
        plt.close()

        # auc_csv_path = os.path.join(out_folder, 'auc_separate.csv')
        # df_auc = pd.DataFrame.from_dict(auc_dict, orient='index')
        # logger.info(f'Save auc csv to {auc_csv_path}')
        # df_auc.to_csv(auc_csv_path)

    def save_5_fold_roc_pr_distribution(self, field_flag, out_folder):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(1, 2)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0])
        gs1 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[1])

        logger.info(f'Run 5 fold LDA on {field_flag}')
        df_field, label_list = self._get_df_field(field_flag)
        prob_list, gt_label_list, projected_list = self._get_5_fold_roc_separate_input(df_field, field_flag)

        ax = plt.subplot(gs0[0])
        for idx_fold in range(5):
            prob = prob_list[idx_fold]
            gt_label = gt_label_list[idx_fold]
            auc_val = round(metrics.roc_auc_score(gt_label, prob), 3)
            fpr, tpr, _ = metrics.roc_curve(gt_label, prob, pos_label=1)
            num_pos_test = np.sum(gt_label)
            plt.plot(fpr, tpr, label=f'Fold: {idx_fold + 1}, # pos val: {num_pos_test}, AUC: {auc_val}')

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC - {field_flag}')
        plt.legend(loc='best')

        ax = plt.subplot(gs0[1])
        for idx_fold in range(5):
            prob = prob_list[idx_fold]
            gt_label = gt_label_list[idx_fold]
            precision, recall, _ = metrics.precision_recall_curve(gt_label, prob)
            auc_val = round(metrics.auc(recall, precision), 3)
            num_pos_test = np.sum(gt_label)
            plt.plot(recall, precision,
                     label=f'Fold: {idx_fold + 1}, # pos val: {num_pos_test}, AUC (PR): {auc_val}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Recall-precision - {field_flag}')
        plt.legend(loc='best')

        for idx_fold in range(5):
            ax = plt.subplot(gs1[idx_fold])
            # prob = prob_list[idx_fold]
            gt_label = gt_label_list[idx_fold]
            projected = projected_list[idx_fold]
            gt_label = np.array(gt_label)
            projected = np.array(projected)
            plt.hist(projected[gt_label == 0], 10, normed=1, facecolor='blue', alpha=0.5)
            y_vals = np.random.normal(0, 0.01, len(gt_label))
            plt.scatter(
                projected,
                gt_label + y_vals,
                c=gt_label,
                cmap='jet',
                alpha=0.3
            )
            plt.ylim(bottom=np.min(gt_label) - 1, top=np.max(gt_label) + 1)
            y_tick_pos = np.arange(2)
            ax.set_yticks(y_tick_pos)
            ax.set_yticklabels(['neg (val)', 'pos (val)'])
            ax.tick_params(which='minor', labelsize=20)
            # plt.title(f'Fold {idx_fold}')

        roc_png_path = os.path.join(out_folder, 'roc_separate.png')
        logger.info(f'Save roc png to {roc_png_path}')
        plt.savefig(roc_png_path)
        plt.close()

        # auc_csv_path = os.path.join(out_folder, 'auc_separate.csv')
        # df_auc = pd.DataFrame.from_dict(auc_dict, orient='index')
        # logger.info(f'Save auc csv to {auc_csv_path}')
        # df_auc.to_csv(auc_csv_path)

    def save_5_fold_roc(self, field_list, out_folder):
        for field_flag in field_list:
            logger.info(f'Run LDA on {field_flag}')
            df_field, label_list = self._get_df_field(field_flag)
            prob, gt_label = self._get_5_fold_roc_input(df_field, field_flag)
            self._roc_data[field_flag] = {
                'label': gt_label,
                'prob': prob
            }

        auc_dict = {}
        plt.figure(figsize=(16, 10))
        for field_flag in self._roc_data:
            y_true = self._roc_data[field_flag]['label']
            y_score = self._roc_data[field_flag]['prob']

            auc_val = round(metrics.roc_auc_score(y_true, y_score), 3)
            auc_dict[field_flag] = {
                'auc': auc_val
            }

            fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
            plt.plot(fpr, tpr, label=f'{field_flag} (AUC: {auc_val})')

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        roc_png_path = os.path.join(out_folder, 'roc.png')
        plt.savefig(roc_png_path)
        plt.close()

        auc_csv_path = os.path.join(out_folder, 'auc.csv')
        df_auc = pd.DataFrame.from_dict(auc_dict, orient='index')
        logger.info(f'Save auc csv to {auc_csv_path}')
        df_auc.to_csv(auc_csv_path)

    def save_roc(self, field_list, out_folder):
        for field_flag in field_list:
            df_field, label_list = self._get_df_field(field_flag)
            prob, gt_label = self._get_roc_input(df_field, field_flag)
            self._roc_data[field_flag] = {
                'label': gt_label,
                'prob': prob
            }

        auc_dict = {}
        plt.figure(figsize=(16, 10))
        for field_flag in self._roc_data:
            y_true = self._roc_data[field_flag]['label']
            y_score = self._roc_data[field_flag]['prob']

            auc_val = round(metrics.roc_auc_score(y_true, y_score), 3)
            auc_dict[field_flag] = {
                'auc': auc_val
            }

            fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
            plt.plot(fpr, tpr, label=f'{field_flag} (AUC: {auc_val})')

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        roc_png_path = os.path.join(out_folder, 'roc.png')
        plt.savefig(roc_png_path)
        plt.close()

        auc_csv_path = os.path.join(out_folder, 'auc.csv')
        df_auc = pd.DataFrame.from_dict(auc_dict, orient='index')
        logger.info(f'Save auc csv to {auc_csv_path}')
        df_auc.to_csv(auc_csv_path)


    def save_auc(self, out_csv):
        pass

    @staticmethod
    def _get_discriminant_metric(x_array, y_array):
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        label_min, label_max = np.min(y_array), np.max(y_array)

        num_class = label_max - label_min + 1
        num_class = int(num_class)
        class_mean_vec = np.zeros(num_class).astype(float)
        class_var_vec = np.zeros(num_class).astype(float)
        for idx_class in range(num_class):
            class_val_vec = x_array[y_array == idx_class]
            class_mean_vec[idx_class] = np.mean(class_val_vec)
            class_var_vec[idx_class] = np.std(class_val_vec)

        abs_dist_vec = np.abs(class_mean_vec[:-1] - class_mean_vec[1:])
        mean_dist = np.sum(abs_dist_vec) / len(abs_dist_vec)
        mean_var = np.sum(class_var_vec) / len(class_var_vec)

        # print(f'dist/var: {mean_dist}/{mean_var}')

        return mean_dist / mean_var

    @staticmethod
    def _count_num_field(df, field_flag, field_val):
        return df[df[field_flag] == field_val].shape[0]

    def _get_df_field(self, field_flag):
        df_field = None
        label_list = []
        if (field_flag == 'copd') | (field_flag == 'COPD'):
            df_field = self._label_df[(self._label_df[field_flag] == 'Yes') | (self._label_df[field_flag] == 'No')]
            df_field = df_field.replace({field_flag: {'Yes': 1, 'No': 0}})
            label_list.append(f'copd:no ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'copd:yes ({self._count_num_field(df_field, field_flag, 1)})')
        elif field_flag == 'Age':
            df_field = self._label_df[(self._label_df[field_flag] < 60) |
                                            (self._label_df[field_flag] > 70)]
            df_field.loc[df_field[field_flag] < 60, field_flag] = 0
            df_field.loc[df_field[field_flag] > 70, field_flag] = 1
            label_list.append(f'Age: <60 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'Age: >70 ({self._count_num_field(df_field, field_flag, 1)})')
        elif (field_flag == 'packyearsreported') | (field_flag == 'Packyear'):
            df_field = self._label_df[
                (self._label_df[field_flag] <= 35) |
                (self._label_df[field_flag] >= 60)]

            df_field.loc[df_field[field_flag] <= 35, field_flag] = 0
            df_field.loc[df_field[field_flag] >= 60, field_flag] = 1

            label_list.append(f'packyear: <35 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'packyear: <60 ({self._count_num_field(df_field, field_flag, 1)})')
        elif field_flag == 'Coronary Artery Calcification':
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
            df_field = self._label_df[(self._label_df[field_flag] < 21) |
                                      (self._label_df[field_flag] > 35)]

            df_field.loc[df_field[field_flag] < 21, field_flag] = 0
            df_field.loc[df_field[field_flag] > 35, field_flag] = 1

            label_list.append(f'BMI: < 21 ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'BMI: > 35 ({self._count_num_field(df_field, field_flag, 1)})')
        elif (field_flag == 'cancer_bengin') | (field_flag == 'Cancer'):
            df_field = self._label_df

            label_list.append(f'non-cancer ({self._count_num_field(df_field, field_flag, 0)})')
            label_list.append(f'cancer ({self._count_num_field(df_field, field_flag, 1)})')
        else:
            raise NotImplementedError

        return df_field, label_list

    def plot_LDA_subspace_2d(self, field_flag, out_png):
        df_field, label_list = self._get_df_field(field_flag)
        self._plot_LDA_subspace_2d(df_field, field_flag, out_png)

    def plot_LDA_subspace_1d(self, field_flag, out_png):
        df_field, label_list = self._get_df_field(field_flag)
        self._plot_LDA_subspace_1d(df_field, label_list, field_flag, out_png)

    def plot_correlation_bar(self, field_flag, out_png):
        df_field, label_list = self._get_df_field(field_flag)
        self._plot_LDA_proj(df_field, field_flag, out_png)

    def plot_copd(self, out_png):
        field_flag = 'copd'
        self.plot_correlation_bar(field_flag, out_png)

    def plot_age(self, out_png):
        field_flag = 'Age'
        self.plot_correlation_bar(field_flag, out_png)

    def plot_packyear(self, out_png):
        field_flag = 'packyearsreported'
        self.plot_correlation_bar(field_flag, out_png)

    def plot_cal(self, out_png):
        field_flag = 'Coronary Artery Calcification'
        self.plot_correlation_bar(field_flag, out_png)

    def plot_bmi(self, out_png):
        field_flag = 'bmi'
        self.plot_correlation_bar(field_flag, out_png)

    @staticmethod
    def add_label_incidental_cancer_flag(in_df):
        cancer_df = in_df[in_df['Cancer'] == 1]
        print(cancer_df.groupby('SubjectID').Time2Diag)

    @staticmethod
    def generate_effective_data_csv(data_array, label_obj, out_csv):
        data_dict = {}
        attribute_list = PlotCorrAnalyzeLDA.attribute_list()
        for data_item in data_array:
            item_dict = {}
            scan_name = data_item['scan_name']

            scan_name_as_record = scan_name
            if not label_obj.check_if_have_record(scan_name):
                logger.info(f'Cannot find record for {scan_name}')
                scan_name_as_record = label_obj.check_nearest_record_for_impute(scan_name)
                if scan_name_as_record is None:
                    continue
                else:
                    logger.info(f'Using nearest record {scan_name_as_record}')

            for attr in attribute_list:
                item_dict[attr] = label_obj.get_value_field(scan_name_as_record, attr)

            item_dict['Cancer'] = item_dict['cancer_bengin']
            item_dict['COPD'] = item_dict['copd']
            item_dict['Packyear'] = item_dict['packyearsreported']
            item_dict['SubjectID'] = label_obj._get_subject_id_from_file_name(scan_name)
            item_dict['ScanDate'] = label_obj._get_date_str_from_file_name(scan_name)
            if item_dict['Cancer'] == 1:
                scan_date_obj = ClinicalDataReaderSPORE._get_date_str_from_file_name(scan_name)
                diag_date_obj = datetime.datetime.strptime(str(int(item_dict['diag_date'])), '%Y%m%d')
                print(str(int(item_dict['diag_date'])))
                print(diag_date_obj)
                item_dict['Time2Diag'] = diag_date_obj - scan_date_obj

            # BMI = mass(lb)/height(inch)^2 * 703
            bmi_val = np.nan
            mass_lb = item_dict['weightpounds']
            height_inch = item_dict['heightinches']
            if (70 < mass_lb < 400) and (40 < height_inch < 90):
                bmi_val = 703 * mass_lb / (height_inch * height_inch)
            item_dict['bmi'] = bmi_val

            for pc_idx in range(20):
                attr_str = PlotCorrAnalyzeLDA.get_pc_str(pc_idx)
                item_dict[attr_str] = data_item['low_dim'][pc_idx]

            data_dict[scan_name] = item_dict

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        PlotCorrAnalyzeLDA.add_label_incidental_cancer_flag(df)

        logger.info(f'Save to csv {out_csv}')
        df.to_csv(out_csv)

    @staticmethod
    def create_class_object_w_data(data_array, label_obj):
        data_dict = {}
        attribute_list = PlotCorrAnalyzeLDA.attribute_list()
        for data_item in data_array:
            item_dict = {}
            scan_name = data_item['scan_name']

            if not label_obj.check_if_have_record(scan_name):
                logger.info(f'Cannot find record for {scan_name}')
                continue

            for attr in attribute_list:
                item_dict[attr] = label_obj.get_value_field(scan_name, attr)

            # BMI = mass(lb)/height(inch)^2 * 703
            bmi_val = np.nan
            mass_lb = item_dict['weightpounds']
            height_inch = item_dict['heightinches']
            if (70 < mass_lb < 400) and (40 < height_inch < 90):
                bmi_val = 703 * mass_lb / (height_inch * height_inch)
            item_dict['bmi'] = bmi_val

            for pc_idx in range(20):
                attr_str = PlotCorrAnalyzeLDA.get_pc_str(pc_idx)
                item_dict[attr_str] = data_item['low_dim'][pc_idx]

            data_dict[scan_name] = item_dict

        df = pd.DataFrame.from_dict(data_dict, orient='index')
        print(df)

        return PlotCorrAnalyzeLDA(df)

    @staticmethod
    def create_class_object_w_csv(csv_path):
        logger.info(f'Load csv data file from {csv_path}')
        df = pd.read_csv(csv_path)
        return PlotCorrAnalyzeLDA(df)

    @staticmethod
    def attribute_list():
        return [
            'Age', 'sex', 'race', 'ctscannermake', 'heightinches',
            'weightpounds', 'packyearsreported', 'copd', 'Coronary Artery Calcification',
            'cancer_bengin', 'diag_date'
        ]

    @staticmethod
    def get_pc_str(idx):
        return f'pc{idx}'


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--in-pca-data-bin', type=str)
    parser.add_argument('--out-png-folder', type=str)
    parser.add_argument('--label-file', type=str)
    parser.add_argument('--data-csv', type=str, default=None)
    # parser.add_argument('--low-dim-data-flag', type=str, default='low_dim')
    args = parser.parse_args()

    out_csv = os.path.join(args.out_png_folder, 'data_full.csv')

    low_dim_array = load_object(args.in_pca_data_bin)
    label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_xlsx(args.label_file)
    PlotCorrAnalyzeLDA.generate_effective_data_csv(low_dim_array, label_obj, out_csv)

    plot_obj = PlotCorrAnalyzeLDA.create_class_object_w_csv(out_csv)

    # plot_obj = None
    # if args.data_csv is not None:
    #     plot_obj = PlotCorrAnalyzeLDA.create_class_object_w_csv(args.data_csv)
    # else:
    #     low_dim_array = load_object(args.in_pca_data_bin)
    #     label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_xlsx(args.label_file)
    #     plot_obj = PlotCorrAnalyzeLDA.create_class_object_w_data(low_dim_array, label_obj)
    #     out_csv = os.path.join(args.out_png_folder, 'data.csv')
    #     plot_obj.save_label_file(out_csv)
    #
    # plot_obj.plot_copd(os.path.join(args.out_png_folder, 'copd.png'))
    # plot_obj.plot_age(os.path.join(args.out_png_folder, 'age.png'))
    # plot_obj.plot_packyear(os.path.join(args.out_png_folder, 'packyear.png'))
    # plot_obj.plot_cal(os.path.join(args.out_png_folder, 'cal.png'))
    # plot_obj.plot_bmi(os.path.join(args.out_png_folder, 'bmi.png'))
    # plot_obj.plot_correlation_bar('cancer_bengin', os.path.join(args.out_png_folder, 'cancer.png'))
    #
    # plot_obj.plot_LDA_subspace_1d(
    #     'copd',
    #     os.path.join(args.out_png_folder, 'copd_LDA_1d.png'))
    # plot_obj.plot_LDA_subspace_1d(
    #     'Age',
    #     os.path.join(args.out_png_folder, 'age_LDA_1d.png')
    # )
    # plot_obj.plot_LDA_subspace_1d(
    #     'packyearsreported',
    #     os.path.join(args.out_png_folder, 'packyear_LDA_1d.png')
    # )
    # plot_obj.plot_LDA_subspace_1d(
    #     'Coronary Artery Calcification',
    #     os.path.join(args.out_png_folder, 'cal_LDA_1d.png'))
    # plot_obj.plot_LDA_subspace_1d(
    #     'bmi',
    #     os.path.join(args.out_png_folder, 'bmi_LDA_1d.png')
    # )
    # plot_obj.plot_LDA_subspace_1d(
    #     'cancer_bengin',
    #     os.path.join(args.out_png_folder, 'cancer_LDA_1d.png')
    # )
    #
    # out_csv = os.path.join(args.out_png_folder, 'metric.csv')
    # plot_obj.save_metric(out_csv)

    # plot_obj.save_roc(
    #     ['Cancer', 'COPD', 'Coronary Artery Calcification'],
    #     args.out_png_folder
    # )

    # plot_obj.save_5_fold_roc_separate(
    #     ['Cancer', 'COPD',
    #      'Coronary Artery Calcification', 'Age',
    #      'Packyear', 'bmi'],
    #     args.out_png_folder
    # )

    # plot_obj.save_5_fold_roc_pr_distribution(
    #     'Cancer',
    #     args.out_png_folder
    # )

if __name__ == '__main__':
    main()