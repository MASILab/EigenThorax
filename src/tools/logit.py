import argparse
from tools.utils import get_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import metrics
import statsmodels.api as sm
from patsy.highlevel import dmatrices
from statsmodels.iolib.summary import Summary, summary_params
from os import path
from tools.statics import hl_test
import pandas as pd
import uncertainties.unumpy as unp
import uncertainties as unc
from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
from tools.data_io import load_object
from tools.cross_validation import get_idx_list_array_n_fold_cross_validation
from tools.data_io import save_object
from tools.PLCOm2012 import get_PLCOm2012_score


logger = get_logger('Logit with cross-validation')


class LogitSingleFold:
    def __init__(self, data_x, data_y, exclude_percentile=95):
        self.exclude_percentile = exclude_percentile
        self._get_x_y_data(data_x, data_y)

    def _get_x_y_data(self, x, y):
        exclude_threshold = np.percentile(x, self.exclude_percentile)
        x = np.array(x)
        y = np.array(y)
        print(f'The {self.exclude_percentile} percentile of x: {exclude_threshold}')
        # print(x > exclude_threshold)
        print(f'Number of excluded scans: {len(x[x > exclude_threshold])}')

        x_use = x[x <= exclude_threshold]
        y_use = y[x <= exclude_threshold]

        self.data = {
            'x': x[x <= exclude_threshold],
            'y': y[x <= exclude_threshold]
        }

        return x_use, y_use

    def fit_all_model(self):
        self.fit_model_intercept()
        self.fit_model_linear()
        self.fit_model_quadratic()
        self.fit_model_third()

    def fit_model_intercept(self):
        logger.info('Run logistic regression with intercept only')

        x = self.data['x']

        Y, X = dmatrices('y ~ 1', self.data)

        self.logit_model_intercept = sm.Logit(Y, X)
        self.logit_result_intercept = self.logit_model_intercept.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_intercept)
        # self.run_HL_test(0)

    def fit_model_linear(self):
        logger.info('Run first order logistic regression')

        x = self.data['x']

        Y, X = dmatrices('y ~ x', self.data)

        self.logit_model_linear = sm.Logit(Y, X)
        self.logit_result_linear = self.logit_model_linear.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_linear)
        self.run_HL_test(1)

    def fit_model_quadratic(self):
        logger.info('Run quadratic logistic regression')

        x = self.data['x']

        Y, X = dmatrices('y ~ x + np.power(x, 2)', self.data)

        self.logit_model_quadratic = sm.Logit(Y, X)
        self.logit_result_quadratic = self.logit_model_quadratic.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_quadratic)
        self.run_HL_test(2)

    def fit_model_third(self):
        logger.info('Run quadratic logistic regression')

        x = self.data['x']

        Y, X = dmatrices('y ~ x + np.power(x, 2) + np.power(x, 3)', self.data)

        self.logit_model_third = sm.Logit(Y, X)
        self.logit_result_third = self.logit_model_third.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_third)
        self.run_HL_test(3)

    def show_regression_result(self, result_obj):
        print(result_obj.summary())
        print(f'Estimated params: {result_obj.params}')
        print(f'AIC: {result_obj.aic}, BIC: {result_obj.bic}')
        print(f'Covariance matrix:')
        print(result_obj.cov_params())

    def run_HL_test(self, order):
        data_used = self.data.copy()

        predicted_prob = self.get_y_series_with_order(data_used['x'], order)

        data_used['prob'] = predicted_prob
        data_df = pd.DataFrame.from_dict(data_used)
        hl_test(data_df, 5, 'y')

    def get_y_series_with_order(self, x_series, order, form_flag='prob'):
        exp_term = None
        num_features = len(x_series)
        if order == 0:
            exp_term = np.zeros((num_features,), dtype=float)
            exp_term[:] = self.logit_result_intercept.params[0]
        elif order == 1:
            x_series_with_intercept = np.zeros((num_features, 2), dtype=float)
            x_series_with_intercept[:, 0] = 1.
            x_series_with_intercept[:, 1] = x_series[:]
            exp_term = np.dot(x_series_with_intercept, self.logit_result_linear.params)
        elif order == 2:
            x_series_with_intercept = np.zeros((num_features, 3), dtype=float)
            x_series_with_intercept[:, 0] = 1.
            x_series_with_intercept[:, 1] = x_series[:]
            x_series_with_intercept[:, 2] = np.power(x_series[:], 2)
            exp_term = np.dot(x_series_with_intercept, self.logit_result_quadratic.params)
        elif order == 3:
            x_series_with_intercept = np.zeros((num_features, 4), dtype=float)
            x_series_with_intercept[:, 0] = 1.
            x_series_with_intercept[:, 1] = x_series[:]
            x_series_with_intercept[:, 2] = np.power(x_series[:], 2)
            x_series_with_intercept[:, 3] = np.power(x_series[:], 3)
            exp_term = np.dot(x_series_with_intercept, self.logit_result_third.params)
        else:
            raise NotImplementedError

        y_series = None
        if form_flag == 'prob':
            y_series = sm.Logit([0], [0]).cdf(exp_term)
        elif form_flag == 'log_odds_ratio':
            y_series = exp_term[:]

        return y_series

    def get_validation_result(self, validation_x, validation_y):
        """
        Get the result on validation set
        :param validation_x:
        :param validation_y:
        :return:
        validation result summary
        """
        validation_summary = []
        for idx_order in range(4):
            predicted_y = self.get_y_series_with_order(validation_x, idx_order, 'prob')
            summary_item = LogitSingleFold.get_validation_statics(validation_y, predicted_y)
            validation_summary.append(summary_item)

        return validation_summary

    @staticmethod
    def get_validation_statics(label, predicted_prob):
        fpr, tpr, _ = metrics.roc_curve(label, predicted_prob, pos_label=1)
        precision, recall, _ = metrics.precision_recall_curve(label, predicted_prob, pos_label=1)
        roc_auc = metrics.roc_auc_score(label, predicted_prob)
        prc_auc = metrics.auc(recall, precision)

        summary_item = {
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'prc_auc': prc_auc,
            'label': label,
            'pred': predicted_prob
        }

        return summary_item

    @staticmethod
    def get_mean_validation_statics_for_cv_array(valid_result_array):
        label = []
        pred = []

        for valid_result in valid_result_array:
            label.append(valid_result['label'])
            pred.append(valid_result['pred'])

        label = np.concatenate(label)
        pred = np.concatenate(pred)

        fpr, tpr, _ = metrics.roc_curve(label, pred, pos_label=1)
        precision, recall, _ = metrics.precision_recall_curve(label, pred, pos_label=1)
        roc_auc = metrics.roc_auc_score(label, pred)
        prc_auc = metrics.auc(recall, precision)

        summary_item = {
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'prc_auc': prc_auc,
            'label': label,
            'pred': pred
        }


class GetGaussianFitSingleFold:
    def __init__(self):
        pass

    def fit_gaussian(self, in_data_matrix):
        """
        Get the gaussian model ready.
        :return:
        """
        num_sample = in_data_matrix.shape[0]
        logger.info(f'MLE with Multivariate Gaussian')
        logger.info(f'Number of samples {num_sample}')
        mu = np.average(in_data_matrix, axis=0)
        x_minus_mu = in_data_matrix - mu
        cov_mat = np.dot(np.transpose(x_minus_mu), x_minus_mu) / num_sample

        self.gaussian_model = multivariate_normal(mean=mu, cov=cov_mat)
        logger.info(f'Done')

        # print(mu)
        # print(cov_mat)

        return mu, cov_mat

    def get_mahalanobis(self, test_data_matrix):
        num_data = test_data_matrix.shape[0]
        logger.info(f'Get m-distance for input data')
        logger.info(f'Number test data: {num_data}')

        mu = self.gaussian_model.mean
        cov = self.gaussian_model.cov
        inv_cov = np.linalg.inv(cov)

        m_dist_array = []
        for idx_data in range(num_data):
            sample_data = test_data_matrix[idx_data]
            sample_m_dist = mahalanobis(sample_data, mu, inv_cov)
            m_dist_array.append(sample_m_dist)

        return m_dist_array


class GetLogitResultCrossValidation:
    def __init__(self, num_fold):
        self.num_fold = num_fold
        self.file_list = None
        self.label_list = None
        self.feature_matrix = None
        self.train_idx_list_fold_array = None
        self.test_idx_list_fold_array = None
        self.gaussian_fit_obj_fold_array = []
        self.logit_model_obj_fold_array = []
        self.validation_result_fold_array = []
        self.PLCOm2012_validation = None

    def load_data_single(self, in_feature_matrix_bin_path, num_pc):
        """
        Load data matrix from single bin file.
        :param in_feature_matrix_bin_path:
        :param num_pc:
        :return:
        """
        logger.info(f'Load bin data file {in_feature_matrix_bin_path}')
        data_obj = load_object(in_feature_matrix_bin_path)
        self.file_list = data_obj['file_list']
        num_pc = int(num_pc)
        self.feature_matrix = data_obj['projected_matrix'][:, :num_pc]

    def load_data_list(self, in_feature_matrix_bin_path_list, num_pc_list):
        """
        Load data matrix from multiple bin files
        :param in_feature_matrix_bin_path_list:
        :param num_pc_list:
        :return:
        """
        pass

    def load_label_file(self, in_csv):
        logger.info(f'Load label file {in_csv}')
        df = pd.read_csv(in_csv, index_col='Scan')
        label_dict = df.to_dict(orient='index')

        # check if have missing label
        missing_list = [file_name for file_name in self.file_list if file_name not in label_dict]
        if len(missing_list) > 0:
            logger.info(f'Missing label:')
            print(missing_list)
            raise NotImplementedError

        self.label_list = np.array([label_dict[file_name]['Cancer'] for file_name in self.file_list])

    def create_cross_validation_folds(self):
        logger.info(f'Create cross-validation folds ({self.num_fold})')
        self.train_idx_list_fold_array, self.test_idx_list_fold_array = \
            get_idx_list_array_n_fold_cross_validation(self.file_list, self.label_list, self.num_fold)

    def get_gaussian_fit_model_fold(self):
        for idx_fold in range(self.num_fold):
            logger.info('')
            logger.info(f'***Gaussian fit with fold {idx_fold}')
            fold_train_data_idx_list = self.train_idx_list_fold_array[idx_fold]
            gaussian_fit_obj = GetGaussianFitSingleFold()
            gaussian_fit_obj.fit_gaussian(self.feature_matrix[fold_train_data_idx_list])
            self.gaussian_fit_obj_fold_array.append(gaussian_fit_obj)

    def get_logit_model_fold(self):
        for idx_fold in range(self.num_fold):
            logger.info('')
            logger.info(f'***Logistic regression with fold {idx_fold}')
            fold_train_data_idx_list = np.array(self.train_idx_list_fold_array[idx_fold])
            fold_gaussian_obj = self.gaussian_fit_obj_fold_array[idx_fold]
            data_x_fold = \
                fold_gaussian_obj.get_mahalanobis(self.feature_matrix[fold_train_data_idx_list])
            data_y_fold = self.label_list[fold_train_data_idx_list]
            logit_obj = LogitSingleFold(data_x_fold, data_y_fold)
            logit_obj.fit_all_model()
            self.logit_model_obj_fold_array.append(logit_obj)

    def get_validation_result(self):
        for idx_fold in range(self.num_fold):
            logger.info('')
            logger.info(f'***Run validation test for fold {idx_fold}')

            fold_gaussian_obj = self.gaussian_fit_obj_fold_array[idx_fold]
            fold_logit_model_obj = self.logit_model_obj_fold_array[idx_fold]

            fold_test_data_idx_list = self.test_idx_list_fold_array[idx_fold]
            test_data_x_fold = fold_gaussian_obj.get_mahalanobis(self.feature_matrix[fold_test_data_idx_list])
            test_data_y_fold = self.label_list[fold_test_data_idx_list]

            validation_result = fold_logit_model_obj.get_validation_result(test_data_x_fold, test_data_y_fold)
            self.validation_result_fold_array.append(validation_result)

    def plot_auc_roc_with_CI(self, out_png):
        fig, ax = plt.subplots(figsize=(18, 12))

        self._plot_auc_roc_with_CI_logit_order(ax, 1, 'Logistic Regression ROC')

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_auc_roc_with_CI_4_order(self, out_png):
        fig, ax = plt.subplots(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.25, hspace=0.2)

        self._plot_auc_roc_with_CI_logit_order(plt.subplot(gs[0]), 0, 'Logistic Regression ROC - Null Model')
        self._plot_auc_roc_with_CI_logit_order(plt.subplot(gs[1]), 1, 'Logistic Regression ROC - P ~ logit(1 + MH)')
        self._plot_auc_roc_with_CI_logit_order(plt.subplot(gs[2]), 2, 'Logistic Regression ROC - P ~ logit(1 + MH + MH^2)')
        self._plot_auc_roc_with_CI_logit_order(plt.subplot(gs[3]), 3, 'Logistic Regression ROC - P ~ logit(1 + MH + MH^2 + MH^3)')

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def get_PLCOm2012_validation_statics(self):
        logger.info('Validation with PLCOm2012')
        PLCOm2012_pred, valid_idx_list = get_PLCOm2012_score(self.file_list)
        valid_label_list = [self.label_list[idx] for idx in valid_idx_list]
        validation_summary = LogitSingleFold.get_validation_statics(valid_label_list, PLCOm2012_pred)
        self.PLCOm2012_validation = validation_summary
        return validation_summary

    def _plot_auc_roc_with_CI_logit_order(self, ax, order_flag, title_str):

        ax.plot([0, 1], [0, 1], linestyle='--', color='b', lw=2, label='No skill', alpha=0.8)

        fpr_array = [valid_result[order_flag]['fpr'] for valid_result in self.validation_result_fold_array]
        tpr_array = [valid_result[order_flag]['tpr'] for valid_result in self.validation_result_fold_array]

        mean_fpr = np.linspace(0, 1, 100)

        mean_tpr, std_tpr = self._get_mean_std_with_interp(fpr_array, tpr_array, mean_fpr)
        mean_tpr[-1] = 1.0
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        auc_array = [valid_result[order_flag]['roc_auc'] for valid_result in self.validation_result_fold_array]
        std_auc = np.std(np.array(auc_array))

        # plot ROC of each fold
        for idx_fold in range(self.num_fold):
            ax.plot(fpr_array[idx_fold], tpr_array[idx_fold], lw=1, label=f'Fold {idx_fold+1} (AUC = {auc_array[idx_fold]:.3f})', alpha=0.3)

        # plot mean
        ax.plot(mean_fpr, mean_tpr, color='r', lw=2, label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')

        # plot baseline
        PLCOm2012_summary = self.PLCOm2012_validation
        PLCOm2012_auc_roc = PLCOm2012_summary['roc_auc']
        ax.plot(PLCOm2012_summary['fpr'], PLCOm2012_summary['tpr'], color='g', lw=2, label=f'PLCOm2012 (AUC = {PLCOm2012_auc_roc:.3f})')

        # plot std
        # ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
        #                 label=r'$\pm$ 1 std. dev.')

        logger.info(f'Plot AUC-ROC ({self.num_fold}-fold cross-validation) for logit model with order {order_flag}')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title_str)

        ax.legend(loc='best')

    def plot_auc_prc_with_CI(self, out_png):
        fix, ax = plt.subplots(figsize=(18, 12))

        self._plot_auc_prc_with_CI_logit_order(ax, 1, 'Logistic Regression - PR-Curve')

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _plot_auc_prc_with_CI_logit_order(self, ax, order_flag, title_str):
        no_skill_val = len(self.label_list[self.label_list==1]) / len(self.label_list)
        ax.plot([0, 1], [no_skill_val, no_skill_val], linestyle='--', color='b', lw=2, label=f'No skill ({no_skill_val:.3f})', alpha=0.8)

        precision_array = [valid_result[order_flag]['precision'] for valid_result in self.validation_result_fold_array]
        recall_array = [valid_result[order_flag]['recall'] for valid_result in self.validation_result_fold_array]

        mean_recall = np.linspace(0, 1, 100)
        mean_precision, std_precision = self._get_mean_std_with_interp(recall_array, precision_array, mean_recall)

        print(mean_precision[:10])

        precision_upper = mean_precision + std_precision
        precision_lower = mean_precision - std_precision

        mean_auc = metrics.auc(mean_precision, mean_recall)
        auc_array = [valid_result[order_flag]['prc_auc'] for valid_result in self.validation_result_fold_array]
        std_auc = np.std(np.array(auc_array))

        # plot PR-Curve of each fold
        for idx_fold in range(self.num_fold):
            ax.plot(recall_array[idx_fold], precision_array[idx_fold], lw=1, label=f'Fold {idx_fold+1} (AUC = {auc_array[idx_fold]:.3f})', alpha=0.3)

        # plot mean
        ax.plot(mean_recall, mean_precision, color='r', lw=2, label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})')

        # plot baseline
        PLCOm2012_summary = self.PLCOm2012_validation
        PLCOm2012_auc_roc = PLCOm2012_summary['prc_auc']
        ax.plot(PLCOm2012_summary['recall'], PLCOm2012_summary['precision'], color='g', lw=2, label=f'PLCOm2012 (AUC = {PLCOm2012_auc_roc:.3f})')

        # plot std
        ax.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        logger.info(f'Plot PR-Curve ({self.num_fold}-fold cross-validation) for logit model with order {order_flag}')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title_str)
        ax.legend(loc='best')

    def _get_mean_std_with_interp(self, x_fold_array, y_fold_array, sample_pts):
        interp_y_list = np.zeros((len(x_fold_array), len(sample_pts)), dtype=float)
        for idx_fold in range(len(x_fold_array)):
            x_fold = x_fold_array[idx_fold]
            y_fold = y_fold_array[idx_fold]
            print(x_fold[:3])
            print(y_fold[:3])
            interp_y_list[idx_fold] = np.interp(sample_pts, x_fold, y_fold)

        mean_y = np.mean(interp_y_list, axis=0)
        std_y = np.std(interp_y_list, axis=0)

        return mean_y, std_y

    def save_validation_result_to_bin(self, out_bin_path):
        save_object(self.validation_result_fold_array, out_bin_path)

    def load_validation_result_from_bin(self, in_bin_path):
        self.validation_result_fold_array = load_object(in_bin_path)
