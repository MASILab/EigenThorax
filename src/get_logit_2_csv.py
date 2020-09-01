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
    parser.add_argument('--in-csv-0-list', nargs='+', type=str)
    parser.add_argument('--in-csv-1-list', nargs='+', type=str)
    parser.add_argument('--label-list', nargs='+', type=str)
    parser.add_argument('--color-list', nargs='+', type=str)
    parser.add_argument('--num-test', type=int)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--out-png', type=str)
    args = parser.parse_args()

    logger.info(f'Run logistic regression')

    num_test = args.num_test
    in_csv_0_list = args.in_csv_0_list
    in_csv_1_list = args.in_csv_1_list
    label_list = args.label_list
    color_list = args.color_list

    fig, ax = plt.subplots(figsize=(20, 7))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0.1, hspace=0.025)
    ax_roc = plt.subplot(gs[0])
    ax_prc = plt.subplot(gs[1])

    no_skill_prc = 28 / 764
    ax_prc.plot([0, 1], [no_skill_prc, no_skill_prc], linestyle='--', label='No Skill')
    ax_roc.plot([0, 1], [0, 1], linestyle='--', label='No Skill')

    for idx_test in range(num_test):
        in_csv_0 = in_csv_0_list[idx_test]
        in_csv_1 = in_csv_1_list[idx_test]
        label_str = label_list[idx_test]
        color_str = color_list[idx_test]

        logger.info(f'Reading {in_csv_0}')
        rvs0 = pd.read_csv(in_csv_0)[args.column_flag].to_numpy()
        logger.info(f'Data length {len(rvs0)}')

        logger.info(f'Reading {in_csv_1}')
        rvs1 = pd.read_csv(in_csv_1)[args.column_flag].to_numpy()
        logger.info(f'Data length {len(rvs1)}')

        # print(rvs0)
        # print(rvs1)
        X = np.zeros((len(rvs0) + len(rvs1), 1), dtype=float)
        X[:len(rvs0), 0] = rvs0[:]
        X[len(rvs0):, 0] = rvs1[:]
        # X = np.concatenate([rvs0, rvs1])
        y = np.zeros((len(rvs0) + len(rvs1),), dtype=int)
        y[:len(rvs0)] = 0
        y[len(rvs0):] = 1

        # print(X.shape)
        # print(y.shape)
        logit_estimator = LogisticRegression(random_state=0).fit(X, y)

        predicted_prob = logit_estimator.predict_proba(X)

        # print(y.shape)
        # print(predicted_prob.shape)
        # print(predicted_prob)
        fpr, tpr, _ = metrics.roc_curve(y, predicted_prob[:, 1], pos_label=1)
        precision, recall, _ = metrics.precision_recall_curve(y, predicted_prob[:, 1], pos_label=1)

        roc_auc = round(metrics.roc_auc_score(y, predicted_prob[:, 1]), 3)
        prc_auc = round(metrics.auc(recall, precision), 3)

        # ax_roc.plot(fpr, tpr, label=f'{label_str} (AUC: {roc_auc})', color=color_str)
        # ax_prc.plot(recall, precision, label=f'{label_str} (AUC: {prc_auc})', color=color_str)
        ax_roc.plot(fpr, tpr, label=f'{label_str} (AUC: {roc_auc})')
        ax_prc.plot(recall, precision, label=f'{label_str} (AUC: {prc_auc})')

    ax_roc.legend(loc='best')
    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title('ROC curve')

    ax_prc.legend(loc='best')
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Recall-Precision curve')

    logger.info(f'Save plot to {args.out_png}')
    plt.savefig(args.out_png, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == '__main__':
    main()