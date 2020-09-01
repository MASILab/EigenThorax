import argparse
from tools.utils import get_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import metrics
import statsmodels.api as sm
from patsy.highlevel import dmatrices


logger = get_logger('Logistic regression, plot')


class GetLogitResult:
    def __init__(self):
        self.logit_model = None
        self.logit_result = None
        self.data = None
        self.view_range_min = 3.
        self.view_range_max = 13.

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

        Y, X = dmatrices('y ~ x', self.data)

        self.logit_model = sm.Logit(Y, X)
        self.logit_result = self.logit_model.fit()

        print(self.logit_result.summary())
        print(f'Estimated params: {self.logit_result.params}')

    def plot_result(self, out_png):
        fig, ax = plt.subplots(figsize=(8, 12))
        gs = gridspec.GridSpec(3, 1)
        gs.update(wspace=0.1, hspace=0.1)

        ax_prob = plt.subplot(gs[0])
        ax_odds = plt.subplot(gs[1])
        ax_log_odds = plt.subplot(gs[2])

        self.plot_probability_curve(ax_prob)
        self.plot_odds_ratio(ax_odds)
        self.plot_log_odds_ratio(ax_log_odds)

        logger.info(f'Save plot to {out_png}')
        fig.tight_layout()
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_probability_curve(self, ax):
        x = self.data['x']
        y = self.data['y']

        y_vals = np.random.normal(0, 0.01, len(y))
        ax.scatter(
            x,
            y + y_vals,
            c=y,
            cmap='jet',
            alpha=0.3
        )

        y_tick_pos = np.arange(2)
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(['All', 'Cancer'])
        ax.set_ylabel('Category')
        ax.tick_params(which='minor', labelsize=20)
        ax.set_xlabel('Mahalanobis distance')

        # plot curve
        # data_min = np.min(x)
        # data_max = np.max(x)
        # data_range = data_max - data_min
        # range_extend_ratio = 0.1
        # view_range_min = (np.min(X) - range_extend_ratio * data_range)
        # view_range_max = (np.max(X) + range_extend_ratio * data_range)
        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        y_series = self.logit_model.cdf(np.dot(x_series_with_intercept, self.logit_result.params))

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Fitted Probability', color=color)
        ax2.plot(
            x_series,
            y_series,
            color=color
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')

    def plot_odds_ratio(self, ax):
        x = self.data['x']
        y = self.data['y']

        y_vals = np.random.normal(0, 0.01, len(y))
        ax.scatter(
            x,
            y + y_vals,
            c=y,
            cmap='jet',
            alpha=0.3
        )

        y_tick_pos = np.arange(2)
        ax.set_yticks(y_tick_pos)
        ax.set_yticklabels(['All', 'Cancer'])
        ax.set_ylabel('Category')
        ax.tick_params(which='minor', labelsize=20)
        ax.set_xlabel('Mahalanobis distance')

        # plot curve
        # data_min = np.min(x)
        # data_max = np.max(x)
        # data_range = data_max - data_min
        # range_extend_ratio = 0.1
        # view_range_min = (np.min(X) - range_extend_ratio * data_range)
        # view_range_max = (np.max(X) + range_extend_ratio * data_range)
        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        # y_series = self.logit_model.cdf(np.dot(x_series_with_intercept, self.logit_result.params))
        y_series = np.exp(np.dot(x_series_with_intercept, self.logit_result.params))

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Fitted odds ratio', color=color)
        ax2.plot(
            x_series,
            y_series,
            color=color
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')

    def plot_log_odds_ratio(self, ax):
        x = self.data['x']
        y = self.data['y']

        # y_vals = np.random.normal(0, 0.01, len(y))
        # ax.scatter(
        #     x,
        #     y + y_vals,
        #     c=y,
        #     cmap='jet',
        #     alpha=0.3
        # )

        # y_tick_pos = np.arange(2)
        # ax.set_yticks(y_tick_pos)
        # ax.set_yticklabels(['All', 'Cancer'])
        # ax.set_ylabel('Category')
        ax.set_ylabel('Count')
        # ax.tick_params(which='minor', labelsize=20)
        ax.set_xlabel('Mahalanobis distance')

        data_array_sequence = [x[y == 0], x[y == 1]]
        color_list = ['blue', 'red']
        label_list = ['All', 'Cancer']
        ax.hist(data_array_sequence, bins=10, color=color_list, label=label_list, alpha=0.5, rwidth=0.9)
        ax.legend(loc=5)
        # ax.hist(x[y == 0], 10, normed=1, facecolor='blue', alpha=0.5)
        # ax.hist(x[y == 1], 10, normed=0.01, facecolor='red', alpha=0.5)

        # plot curve
        # data_min = np.min(x)
        # data_max = np.max(x)
        # data_range = data_max - data_min
        # range_extend_ratio = 0.1
        # view_range_min = (np.min(X) - range_extend_ratio * data_range)
        # view_range_max = (np.max(X) + range_extend_ratio * data_range)
        view_range_min = self.view_range_min
        view_range_max = self.view_range_max
        step_size = (view_range_max - view_range_min) / 100
        x_series = np.arange(start=view_range_min,
                             stop=view_range_max,
                             step=step_size)
        x_series_with_intercept = np.zeros((100, 2), dtype=float)
        x_series_with_intercept[:, 0] = 1.
        x_series_with_intercept[:, 1] = x_series[:]
        # y_series = self.logit_model.cdf(np.dot(x_series_with_intercept, self.logit_result.params))
        y_series = np.dot(x_series_with_intercept, self.logit_result.params)

        ax2 = ax.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Fitted Logit (Log odds ratio)', color=color)
        ax2.plot(
            x_series,
            y_series,
            color=color
        )
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.grid(b=True, color=color, linestyle='--')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv-1', type=str)
    parser.add_argument('--in-csv-2', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    logit_obj = GetLogitResult()
    logit_obj.fit_model(args.in_csv_1, args.in_csv_2, args.column_flag)
    logit_obj.plot_result(args.out_png)


if __name__ == '__main__':
    main()