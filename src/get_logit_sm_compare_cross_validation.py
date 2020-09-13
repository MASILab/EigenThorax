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


logger = get_logger('Logistic regression, plot')


class GetLogitResultCrossValidation:
    def __init__(self):
        self.logit_model_linear = None
        self.logit_result_linear = None
        self.logit_model_quadratic = None
        self.logit_result_quadratic = None
        self.logit_model_third = None
        self.logit_result_third = None
        self.data = None
        self.view_range_min = 1.
        self.view_range_max = 13.
        self.band_sample_size = 50
        self.color_val_ax = 'tab:orange'
        self.hist_num_bins = 10

    def fit_model(self, in_csv_positive, in_csv_negative, column_flag):
        x, y = self.get_x_y_data(in_csv_positive, in_csv_negative, column_flag)

        self.fit_model_intercept(x, y)
        self.fit_model_linear(x, y)
        self.fit_model_quadratic(x, y)
        self.fit_model_third(x, y)

    def get_x_y_data(self, in_csv_positive, in_csv_negative, column_flag):

        logger.info(f'Reading positive data from {in_csv_positive}')
        rvs_pos = pd.read_csv(in_csv_positive)[column_flag].to_numpy()
        logger.info(f'Reading negative data from {in_csv_negative}')
        rvs_neg = pd.read_csv(in_csv_negative)[column_flag].to_numpy()

        x = np.zeros((len(rvs_pos) + len(rvs_neg),), dtype=float)
        x[:len(rvs_pos)] = rvs_pos[:]
        x[len(rvs_pos):] = rvs_neg[:]
        y = np.zeros((len(rvs_pos) + len(rvs_neg),), dtype=int)
        y[:len(rvs_pos)] = 1
        y[len(rvs_pos):] = 0

        exclude_threshold = np.percentile(x, 97.5)
        print(f'The 97.5 percentile of x: {exclude_threshold}')
        print(f'Number of scans: {len(x[x > exclude_threshold])}')

        self.data = {
            'x': x[x <= exclude_threshold],
            'y': y[x <= exclude_threshold]
        }

        return x, y

    def fit_model_intercept(self, x, y):
        logger.info('Run logistic regression with intercept only')

        self.data = {
            'x': x,
            'y': y
        }

        Y, X = dmatrices('y ~ 1', self.data)

        self.logit_model_intercept = sm.Logit(Y, X)
        self.logit_result_intercept = self.logit_model_intercept.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_intercept)
        # self.run_HL_test(0)

    def fit_model_linear(self, x, y):
        logger.info('Run first order logistic regression')

        # x, y = self.get_x_y_data(in_csv_1, in_csv_2, column_flag)

        self.data = {
            'x': x,
            'y': y
        }

        Y, X = dmatrices('y ~ x', self.data)

        self.logit_model_linear = sm.Logit(Y, X)
        self.logit_result_linear = self.logit_model_linear.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_linear)
        self.run_HL_test(1)

    def fit_model_quadratic(self, x, y):
        logger.info('Run quadratic logistic regression')

        self.data = {
            'x': x,
            'y': y
        }

        Y, X = dmatrices('y ~ x + np.power(x, 2)', self.data)

        self.logit_model_quadratic = sm.Logit(Y, X)
        self.logit_result_quadratic = self.logit_model_quadratic.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        self.show_regression_result(self.logit_result_quadratic)
        self.run_HL_test(2)

    def fit_model_third(self, x, y):
        logger.info('Run quadratic logistic regression')

        self.data = {
            'x': x,
            'y': y
        }

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

    def plot_result(self, out_png):
        fig, ax = plt.subplots(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.25, hspace=0.2)

        ax_prob = plt.subplot(gs[0])
        ax_odds = plt.subplot(gs[1])
        ax_log_odds = plt.subplot(gs[2])
        # ax_prob_extended_range = plt.subplot(gs[3])

        # self.view_range_min = 0.
        # self.view_range_max = 50.
        # self.plot_probability_curve(ax_prob_extended_range)
        # self.view_range_min = np.min(self.data['x'])
        # self.view_range_max = np.max(self.data['x'])

        self.plot_probability_curve(ax_prob)
        self.plot_odds_ratio(ax_odds)
        self.plot_log_odds_ratio(ax_log_odds)

        logger.info(f'Save plot to {out_png}')
        fig.tight_layout()
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_result_with_density_estimate(self, out_png):
        fig, ax = plt.subplots(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.25, hspace=0.2)

        ax_prob = plt.subplot(gs[0])
        ax_log_odds = plt.subplot(gs[1])
        ax_prob_coarse = plt.subplot(gs[2])
        ax_log_odds_coarse = plt.subplot(gs[3])

        self.hist_num_bins = 10
        self.plot_probability_curve(ax_prob)
        self.plot_log_odds_ratio(ax_log_odds)
        self.hist_num_bins = 5
        self.plot_probability_curve(ax_prob_coarse)
        self.plot_log_odds_ratio(ax_log_odds_coarse)

        logger.info(f'Save plot to {out_png}')
        fig.tight_layout()
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_result_prob_only(self, out_png):
        fig, ax = plt.subplots(figsize=(8, 5))

        self.plot_probability_curve(ax)

        logger.info(f'Save plot to {out_png}')
        fig.tight_layout()
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_order_compare_result(self, out_png):
        fig, ax = plt.subplots(figsize=(24, 8))
        gs = gridspec.GridSpec(1, 2)
        gs.update(wspace=0.25, hspace=0.2)

        ax_prob = plt.subplot(gs[0])
        ax_odds = plt.subplot(gs[1])

        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)

        # Plot prob
        hist_info = self.plot_hist(ax_prob)
        ax2_prob = ax_prob.twinx()
        ax2_prob.set_ylabel('Probability')
        self.plot_hist_bin_bubble_plot(hist_info, ax2_prob, 'prob')
        self.plot_data_curve(
            ax2_prob,
            x_series,
            self.get_y_series_with_order(x_series, 0, 'prob'),
            'P ~ Logit(1)'
        )
        self.plot_data_curve(
            ax2_prob,
            x_series,
            self.get_y_series_with_order(x_series, 1, 'prob'),
            'P ~ Logit(1 + MH)'
        )
        self.plot_data_curve(
            ax2_prob,
            x_series,
            self.get_y_series_with_order(x_series, 2, 'prob'),
            'P ~ Logit(1 + MH + HM^2)'
        )
        self.plot_data_curve(
            ax2_prob,
            x_series,
            self.get_y_series_with_order(x_series, 3, 'prob'),
            'P ~ Logit(1 + MH + HM^2 + HM^3)'
        )
        ax2_prob.legend(loc=1)
        ax2_prob.tick_params(axis='y')
        ax2_prob.grid(b=True, linestyle='--')
        ax2_prob.set_title('Logistic regression result (probability)')

        # Plot log_odds_ratio
        hist_info = self.plot_hist(ax_odds)
        ax2_odds = ax_odds.twinx()
        ax2_odds.set_ylabel('Log odds ratio (Logit)')
        self.plot_hist_bin_bubble_plot(hist_info, ax2_odds, 'log_odds_ratio')
        self.plot_data_curve(
            ax2_odds,
            x_series,
            self.get_y_series_with_order(x_series, 0, 'log_odds_ratio'),
            'P ~ Logit(1)'
        )
        self.plot_data_curve(
            ax2_odds,
            x_series,
            self.get_y_series_with_order(x_series, 1, 'log_odds_ratio'),
            'P ~ Logit(1 + MH)'
        )
        self.plot_data_curve(
            ax2_odds,
            x_series,
            self.get_y_series_with_order(x_series, 2, 'log_odds_ratio'),
            'P ~ Logit(1 + MH + HM^2)'
        )
        self.plot_data_curve(
            ax2_odds,
            x_series,
            self.get_y_series_with_order(x_series, 3, 'log_odds_ratio'),
            'P ~ Logit(1 + MH + HM^2 + HM^3)'
        )
        ax2_odds.legend(loc=1)
        ax2_odds.tick_params(axis='y')
        ax2_odds.grid(b=True, linestyle='--')
        ax2_odds.set_title('Logistic regression result (log odds ratio)')
        ax2_odds.set_ylim(-6, -1.3)

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def get_title_for_order(self, order):
        if order == 0:
            return 'null model: p ~ logit(1)'
        elif order == 1:
            return 'linear model: p ~ logit(1+MH)'
        elif order == 2:
            return 'quadratic model: p ~ logit(1 + MH + MH^2)'
        elif order == 3:
            return 'cubic model: p ~ logit(1 + MH + MH^2 + MH^3)'
        else:
            raise NotImplementedError

    def plot_CI_band_for_order(self, order, out_png):

        result_obj = self.get_result_order(order)

        fig, ax = plt.subplots(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 2)
        gs.update(wspace=0.25, hspace=0.2)

        ax_prob = plt.subplot(gs[0])
        ax_odds = plt.subplot(gs[1])

        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)

        # Plot prob
        hist_info = self.plot_hist(ax_prob)
        ax2_prob = ax_prob.twinx()
        ax2_prob.set_ylabel('Probability')
        self.plot_hist_bin_bubble_plot(hist_info, ax2_prob, 'prob')
        self.plot_confidence_band_with_cov(
            ax2_prob,
            x_series,
            result_obj,
            order,
            'prob'
        )
        ax2_prob.plot(
            x_series,
            self.get_y_series_with_order(x_series, order, 'prob'),
            label='Predicted',
            color=self.color_val_ax
        )
        ax2_prob.legend(loc=1)
        ax2_prob.tick_params(axis='y')
        ax2_prob.grid(b=True, linestyle='--')
        ax2_prob.set_title(f'Logistic regression result in probability (cancer) scale\n {self.get_title_for_order(order)}')
        ax2_prob.set_ylim(-0.01, 0.17)

        # Plot Log odds ratio
        hist_info = self.plot_hist(ax_odds)
        ax2_odds = ax_odds.twinx()
        ax2_odds.set_ylabel('Log odds ratio (Logit)')
        self.plot_hist_bin_bubble_plot(hist_info, ax2_odds, 'log_odds_ratio')
        self.plot_confidence_band_with_cov(
            ax2_odds,
            x_series,
            result_obj,
            order,
            'log_odds_ratio'
        )
        ax2_odds.plot(
            x_series,
            self.get_y_series_with_order(x_series, order, 'log_odds_ratio'),
            label='Predicted',
            color=self.color_val_ax
        )
        ax2_odds.legend(loc=1)
        ax2_odds.tick_params(axis='y')
        ax2_odds.grid(b=True, linestyle='--')
        ax2_odds.set_title(f'Logistic regression result in log odds ratio (cancer / non-cancer) scale\n {self.get_title_for_order(order)}')
        ax2_odds.set_ylim(-5, -1)

        logger.info(f'Save plot to {out_png}')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

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

    def plot_probability_curve(self, ax):
        hist_info = self.plot_hist(ax)

        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        y_series = self.logit_model_linear.cdf(np.dot(x_series_with_intercept, self.logit_result_linear.params))

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Probability [P]', color=color)
        # self.plot_confidence_band(ax2, x_series, 'prob')
        self.plot_hist_bin_bubble_plot(hist_info, ax2, 'prob')
        ax2.plot(
            x_series,
            y_series,
            color=color,
            label='Predicted'
        )
        ax2.legend(loc=1)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')
        ax2.set_title('Logistic regression: P ~ logit(1 + MH)')

    def plot_odds_ratio(self, ax):
        hist_info = self.plot_hist(ax)

        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        y_series = np.exp(np.dot(x_series_with_intercept, self.logit_result_linear.params))

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Odds ratio [P / (1 - P)]', color=color)
        # self.plot_confidence_band(ax2, x_series, 'odds_ratio')
        self.plot_hist_bin_bubble_plot(hist_info, ax2, 'odds_ratio')
        ax2.plot(
            x_series,
            y_series,
            color=color
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')

    def plot_log_odds_ratio(self, ax):
        hist_info = self.plot_hist(ax)

        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        y_series = np.dot(x_series_with_intercept, self.logit_result_linear.params)

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel(' Log odds ratio [Log(P / (1 - P))]', color=color)

        # self.plot_confidence_band(ax2, x_series, 'log_odds_ratio')
        self.plot_hist_bin_bubble_plot(hist_info, ax2, 'log_odds_ratio')
        ax2.plot(
            x_series,
            y_series,
            color=color
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')

    def plot_hist(self, ax):
        logger.info(f'Plot histogram for cancer / non-cancer distribution')
        x = self.data['x']
        y = self.data['y']

        ax.set_ylabel('Count')
        ax.set_xlabel('Mahalanobis distance')

        # data_array_sequence = [x[y == 0], x[y == 1]]
        # color_list = ['blue', 'red']
        # label_list = ['Non-cancer', 'Cancer']
        data_array_sequence = [x[y == 1], x[y == 0]]
        color_list = ['red', 'blue']
        label_list = ['Cancer', 'Non-cancer']
        hist_info = ax.hist(
            data_array_sequence,
            bins=self.hist_num_bins,
            color=color_list,
            label=label_list,
            stacked=True,
            alpha=0.5, rwidth=0.8)
        print(hist_info)
        ax.legend(loc=2)

        return hist_info

    def plot_hist_bin_bubble_plot(self, hist_info, ax, data_flag):
        hist_value_array, hist_bin_boundary_array, hist_patches = hist_info

        x = (hist_bin_boundary_array[1:] + hist_bin_boundary_array[:-1]) / 2.
        y = None
        dist_pos = hist_value_array[0]
        dist_neg = hist_value_array[1]
        dist_total = dist_pos + dist_neg
        p = dist_pos / dist_total.astype(float)
        if data_flag == 'prob':
            y = p
        elif data_flag == 'odds_ratio':
            y = p / (1. - p)
        elif data_flag == 'log_odds_ratio':
            y = np.log(p / (1. - p))
        else:
            raise NotImplementedError

        s = dist_total

        ax.scatter(x, y,
                   s=3*s,
                   c=self.color_val_ax,
                   alpha=0.5,
                   edgecolors=self.color_val_ax,
                   label=f'Observed ({len(hist_bin_boundary_array) - 1} bins)')
        # ax.scatter(x, y, c=self.color_val_ax)

    def plot_data_curve(self, ax, x_series, y_series, label_str):
        ax.plot(
            x_series,
            y_series,
            label=label_str
        )

    def get_result_order(self, order):
        if order == 0:
            return self.logit_result_intercept
        elif order == 1:
            return self.logit_result_linear
        elif order == 2:
            return self.logit_result_quadratic
        elif order == 3:
            return self.logit_result_third
        else:
            raise NotImplementedError

    def plot_confidence_band_with_cov(self, ax, x_series, result_obj, order, scale_flag='prob'):
        logger.info(f'Plot confidence interval for {order}, {scale_flag}')

        params, cov_mat = self.get_param_cov_result(result_obj)

        param_list_with_cov = list(unc.correlated_values(params, cov_mat))

        intercept_term = np.zeros((len(x_series),), dtype=float)
        intercept_term[:] = 1.

        # Get exp term
        exp_term = None
        if order == 0:
            exp_term = param_list_with_cov[0] * intercept_term
        elif order == 1:
            exp_term = \
                param_list_with_cov[0] * intercept_term + \
                param_list_with_cov[1] * x_series
        elif order == 2:
            exp_term = \
                param_list_with_cov[0] * intercept_term + \
                param_list_with_cov[1] * x_series + \
                param_list_with_cov[2] * np.power(x_series, 2)
        elif order == 3:
            exp_term = \
                param_list_with_cov[0] * intercept_term + \
                param_list_with_cov[1] * x_series + \
                param_list_with_cov[2] * np.power(x_series, 2) + \
                param_list_with_cov[3] * np.power(x_series, 3)

        # Get the 95% upper lower bound of the exp_term. Assume that the distribution is Guassian in log odds ratio space.
        exp_nom = unp.nominal_values(exp_term)
        exp_std = unp.std_devs(exp_term)

        exp_upper = exp_nom + 1.96 * exp_std
        exp_lower = exp_nom - 1.96 * exp_std

        # transform the upper lower bound to requested scale, e.g. prob
        plot_upper = None
        plot_lower = None
        if scale_flag == 'prob':
            plot_upper = sm.Logit([0], [0]).cdf(exp_upper)
            plot_lower = sm.Logit([0], [0]).cdf(exp_lower)
        elif scale_flag == 'log_odds_ratio':
            plot_upper = exp_upper
            plot_lower = exp_lower
        else:
            raise NotImplementedError

        ax.plot(
            x_series,
            plot_upper,
            color=self.color_val_ax,
            linestyle='dashed',
            label='95% Confidence Region'
        )
        ax.plot(
            x_series,
            plot_lower,
            color=self.color_val_ax,
            linestyle='dashed'
        )
        ax.fill_between(
            x_series,
            plot_upper,
            plot_lower,
            facecolor=self.color_val_ax,
            alpha=0.3
        )


    def plot_confidence_band(self, ax, x_series, data_flag):
        logger.info(f'Start to calculate odds ratio confidence band of {data_flag}')

        intercept_range, slope_range = self.get_confidence_interval()

        intercept_series = np.arange(
            start=intercept_range[0],
            stop=intercept_range[1],
            step=(intercept_range[1] - intercept_range[0]) / self.band_sample_size
        )

        slope_series = np.arange(
            start=slope_range[0],
            stop=slope_range[1],
            step=(slope_range[1] - slope_range[0]) / self.band_sample_size
        )

        confidence_band_data = np.zeros((
            self.band_sample_size,
            self.band_sample_size,
            len(x_series)
        ), dtype=float)

        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]

        for idx_intercept in range(len(intercept_series)):
            val_intercept = intercept_series[idx_intercept]
            for idx_slope in range(len(slope_series)):
                val_slope = slope_series[idx_slope]
                y_series = self.get_y_series_data_flag(
                    x_series_with_intercept,
                    np.array([val_intercept, val_slope]),
                    data_flag
                )
                confidence_band_data[idx_intercept, idx_slope, :] = y_series[:]
                ax.plot(
                    x_series,
                    y_series,
                    color=self.color_val_ax,
                    alpha=0.015
                )

        band_min = np.zeros((len(x_series),), dtype=float)
        band_max = np.zeros((len(x_series),), dtype=float)

        for idx_band in range(len(x_series)):
            band_min[idx_band] = np.min(confidence_band_data[:, :, idx_band])
            band_max[idx_band] = np.max(confidence_band_data[:, :, idx_band])

        ax.plot(
            x_series,
            band_min,
            color=self.color_val_ax,
            linestyle='dashed'
        )
        ax.plot(
            x_series,
            band_max,
            color=self.color_val_ax,
            linestyle='dashed'
        )

    def get_confidence_band(self, intercept_range, slope_range, x_series, data_flag):
        logger.info(f'Start to calculate odds ratio confidence band of {data_flag}')

        intercept_series = np.arange(
            start=intercept_range[0],
            stop=intercept_range[1],
            step=(intercept_range[1] - intercept_range[0]) / self.band_sample_size
        )

        slope_series = np.arange(
            start=slope_range[0],
            stop=slope_range[1],
            step=(slope_range[1] - slope_range[0]) / self.band_sample_size
        )

        confidence_band_data = np.zeros((
            self.band_sample_size,
            self.band_sample_size,
            len(x_series)
        ), dtype=float)

        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]

        for idx_intercept in range(len(intercept_series)):
            val_intercept = intercept_series[idx_intercept]
            for idx_slope in range(len(slope_series)):
                val_slope = slope_series[idx_slope]
                y_series = self.get_y_series_data_flag(
                    x_series_with_intercept,
                    np.array([val_intercept, val_slope]),
                    data_flag
                )
                confidence_band_data[idx_intercept, idx_slope, :] = y_series[:]

        band_min = np.zeros((len(x_series),), dtype=float)
        band_max = np.zeros((len(x_series),), dtype=float)

        for idx_band in range(len(x_series)):
            band_min[idx_band] = np.min(confidence_band_data[:, :, idx_band])
            band_max[idx_band] = np.max(confidence_band_data[:, :, idx_band])

        return band_min, band_max

    def get_y_series_data_flag(self, x_series_with_intercept, param_list, data_flag):
        y_series = None

        if data_flag == 'log_odds_ratio':
            y_series = np.dot(x_series_with_intercept, param_list)
        elif data_flag == 'odds_ratio':
            y_series = np.exp(np.dot(x_series_with_intercept, param_list))
        elif data_flag == 'prob':
            y_series = self.logit_model_linear.cdf(np.dot(x_series_with_intercept, param_list))
        else:
            raise NotImplementedError

        return y_series

    def get_parameter_summary(self):
        print(summary_params(self.logit_result_linear).data)

    def get_confidence_interval(self):
        summary_data = summary_params(self.logit_result_linear).data
        intercept_range = [float(summary_data[1][5]), float(summary_data[1][6])]
        slope_range = [float(summary_data[2][5]), float(summary_data[2][6])]

        # print(intercept_range)
        # print(slope_range)

        return intercept_range, slope_range

    def get_param_cov_result(self, result_obj):
        return result_obj.params, result_obj.cov_params()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv-pos', type=str)
    parser.add_argument('--in-csv-neg', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--out-folder', type=str)
    args = parser.parse_args()

    logit_obj = GetLogitResult()
    logit_obj.fit_model(args.in_csv_pos, args.in_csv_neg, args.column_flag)

    # logit_obj.plot_result_with_density_estimate(args.out_png)
    # logit_obj.plot_result(args.out_png)
    # logit_obj.plot_result_prob_only(args.out_png)
    # logit_obj.get_parameter_summary()
    # logit_obj.get_confidence_interval()

    # out_png_compare_result = path.join(args.out_folder, 'order_compare.png')
    # logit_obj.plot_order_compare_result(out_png_compare_result)

    out_png_CI_band_order_0 = path.join(args.out_folder, 'CI_band_order_0.png')
    logit_obj.plot_CI_band_for_order(0, out_png_CI_band_order_0)

    out_png_CI_band_order_1 = path.join(args.out_folder, 'CI_band_order_1.png')
    logit_obj.plot_CI_band_for_order(1, out_png_CI_band_order_1)

    out_png_CI_band_order_2 = path.join(args.out_folder, 'CI_band_order_2.png')
    logit_obj.plot_CI_band_for_order(2, out_png_CI_band_order_2)

    out_png_CI_band_order_3 = path.join(args.out_folder, 'CI_band_order_3.png')
    logit_obj.plot_CI_band_for_order(3, out_png_CI_band_order_3)


if __name__ == '__main__':
    main()