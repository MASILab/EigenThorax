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


logger = get_logger('Logistic regression, plot')


class GetLogitResult:
    def __init__(self):
        self.logit_model = None
        self.logit_result = None
        self.data = None
        self.view_range_min = 1.
        self.view_range_max = 13.
        self.band_sample_size = 50
        self.color_val_ax = 'tab:orange'
        self.hist_num_bins = 10

    def fit_model(self, in_csv_1, in_csv_2, column_flag):
        logger.info(f'Run logistic regression')

        logger.info(f'Reading {in_csv_1}')
        rvs1 = pd.read_csv(in_csv_1)[column_flag].to_numpy()
        logger.info(f'Data length {len(rvs1)}')

        logger.info(f'Reading {in_csv_2}')
        rvs2 = pd.read_csv(in_csv_2)[column_flag].to_numpy()
        logger.info(f'Data length {len(rvs2)}')

        x = np.zeros((len(rvs1) + len(rvs2),), dtype=float)
        x[:len(rvs1)] = rvs1[:]
        x[len(rvs1):] = rvs2[:]
        y = np.zeros((len(rvs1) + len(rvs2),), dtype=int)
        y[:len(rvs1)] = 0
        y[len(rvs1):] = 1

        self.data = {
            'x': x,
            'y': y
        }

        Y, X = dmatrices('y ~ x + np.power(x, 2)', self.data)

        self.logit_model = sm.Logit(Y, X)
        self.logit_result = self.logit_model.fit()

        data_range = np.max(x) - np.min(x)
        self.view_range_min = np.min(x) - 0.05 * data_range
        self.view_range_max = np.max(x) + 0.05 * data_range

        print(self.logit_result.summary())
        print(f'Estimated params: {self.logit_result.params}')

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

    def get_x_y_series(self, data_flag):
        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 3), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        x_series_with_intercept[:, 2] = np.power(x_series, 2)
        y_series = None
        p = self.logit_model.cdf(np.dot(x_series_with_intercept, self.logit_result.params))
        if data_flag == 'prob':
            y_series = p
        elif data_flag == 'odds_ratio':
            y_series = p / (1. - p)
        elif data_flag == 'log_odds_ratio':
            y_series = np.log(p / (1. - p))
        else:
            raise NotImplementedError

        return x_series, y_series

    def plot_probability_curve(self, ax):
        hist_info = self.plot_hist(ax)
        x_series, y_series = self.get_x_y_series('prob')

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Probability [P]', color=color)
        # self.plot_confidence_band(ax2, x_series, 'prob')
        self.plot_hist_bin_bubble_plot(hist_info, ax2, 'prob')
        ax2.plot(
            x_series,
            y_series,
            color=color
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')

    def plot_odds_ratio(self, ax):
        hist_info = self.plot_hist(ax)
        x_series, y_series = self.get_x_y_series('odds_ratio')

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
        x_series, y_series = self.get_x_y_series('log_odds_ratio')

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
        x = self.data['x']
        y = self.data['y']

        ax.set_ylabel('Count')
        ax.set_xlabel('Mahalanobis distance')

        data_array_sequence = [x[y == 0], x[y == 1]]
        color_list = ['blue', 'red']
        label_list = ['All', 'Cancer']
        hist_info = ax.hist(
            data_array_sequence,
            bins=self.hist_num_bins,
            color=color_list,
            label=label_list,
            alpha=0.5, rwidth=0.9)
        print(hist_info)
        ax.legend(loc=2)

        return hist_info

    def plot_hist_bin_bubble_plot(self, hist_info, ax, data_flag):
        hist_value_array, hist_bin_boundary_array, hist_patches = hist_info

        x = (hist_bin_boundary_array[1:] + hist_bin_boundary_array[:-1]) / 2.
        y = None
        p = hist_value_array[1].astype(float) / hist_value_array[0].astype(float)
        if data_flag == 'prob':
            y = p
        elif data_flag == 'odds_ratio':
            y = p / (1. - p)
        elif data_flag == 'log_odds_ratio':
            y = np.log(p / (1. - p))
        else:
            raise NotImplementedError

        s = hist_value_array[0]

        ax.scatter(x, y, s=5*s, c=self.color_val_ax, alpha=0.5, edgecolors=self.color_val_ax)
        # ax.scatter(x, y, c=self.color_val_ax)

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
            y_series = self.logit_model.cdf(np.dot(x_series_with_intercept, param_list))
        else:
            raise NotImplementedError

        return y_series

    def get_parameter_summary(self):
        print(summary_params(self.logit_result).data)

    def get_confidence_interval(self):
        summary_data = summary_params(self.logit_result).data
        intercept_range = [float(summary_data[1][5]), float(summary_data[1][6])]
        slope_range = [float(summary_data[2][5]), float(summary_data[2][6])]

        # print(intercept_range)
        # print(slope_range)

        return intercept_range, slope_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv-1', type=str)
    parser.add_argument('--in-csv-2', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    logit_obj = GetLogitResult()
    logit_obj.fit_model(args.in_csv_1, args.in_csv_2, args.column_flag)
    logit_obj.plot_result_with_density_estimate(args.out_png)
    # logit_obj.plot_result(args.out_png)
    # logit_obj.plot_result_prob_only(args.out_png)
    # logit_obj.get_parameter_summary()
    # logit_obj.get_confidence_interval()


if __name__ == '__main__':
    main()