import argparse
from tools.utils import get_logger
import pandas as pd
from scipy.stats import kstest, ttest_ind, mannwhitneyu, ks_2samp

logger = get_logger('iid test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv-1', type=str)
    parser.add_argument('--in-csv-2', type=str)
    parser.add_argument('--column-flag', type=str)
    parser.add_argument('--method', type=str, default='t-test')
    args = parser.parse_args()

    logger.info(f'Compare the two distributions using method [{args.method}]')

    logger.info(f'Reading {args.in_csv_1}')
    rvs1 = pd.read_csv(args.in_csv_1)[args.column_flag].to_numpy()
    logger.info(f'Data length {len(rvs1)}')

    logger.info(f'Reading {args.in_csv_2}')
    rvs2 = pd.read_csv(args.in_csv_2)[args.column_flag].to_numpy()
    logger.info(f'Data length {len(rvs2)}')

    statics_val = None
    if args.method == 't-test':
        statics_val = ttest_ind(rvs1, rvs2)
        logger.info(f'Run t-test: {statics_val}')
    elif args.method == 'Mann-Whitney':
        statics_val = mannwhitneyu(rvs1, rvs2, alternative='less')
        logger.info(f'Run Mann-Whitney U test: {statics_val}')
    elif args.method == 'kstest':
        statics_val = ks_2samp(rvs1, rvs2, alternative='greater')
        logger.info(f'Run ks-test: {statics_val}')
    else:
        logger.info('Error, cannot find the computing method.')
        raise NotImplementedError

if __name__ == '__main__':
    main()