import argparse
from tools.utils import get_logger
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn import metrics

logger = get_logger('Logistic regression, plot')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv-1', type=str)
    parser.add_argument('--in-csv-2', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    logger.info(f'Run logistic regression')

    logger.info(f'Reading {args.in_csv_1}')
    rvs1 = pd.read_csv(args.in_csv_1)[args.column_flag].to_numpy()
    logger.info(f'Data length {len(rvs1)}')

    logger.info(f'Reading {args.in_csv_2}')
    rvs2 = pd.read_csv(args.in_csv_2)[args.column_flag].to_numpy()
    logger.info(f'Data length {len(rvs2)}')

    fig, ax = plt.subplots(figsize=(8, 4))

    X = np.zeros((len(rvs1) + len(rvs2), 1), dtype=float)
    X[:len(rvs1), 0] = rvs1[:]
    X[len(rvs1):, 0] = rvs2[:]
    y = np.zeros((len(rvs1) + len(rvs2),), dtype=float)
    y[:len(rvs1)] = 0
    y[len(rvs1):] = 1

    logit_estimator = LogisticRegression(random_state=0).fit(X, y)

    # Plot scatters
    y_vals = np.random.normal(0, 0.01, len(y))
    ax.scatter(
        X[:, 0],
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

    # Plot curve
    x_series = np.arange(start=np.min(X), stop=np.max(X), step=(np.max(X)-np.min(X))/100)
    x_series = np.reshape(x_series, (-1, 1))
    y_series = logit_estimator.predict_proba(x_series)

    ax2 = ax.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Predicted Probability (Risk)', color=color)
    ax2.plot(
        x_series[:, 0],
        y_series[:, 1],
        color=color
    )
    ax2.tick_params(axis='y', labelcolor=color)

    logger.info(f'Save plot to {args.out_png}')
    fig.tight_layout()
    plt.savefig(args.out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    main()