import argparse
from tools.utils import get_logger
import pandas as pd
from scipy.stats import kstest, ttest_ind

logger = get_logger('iid test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv-1', type=str)
    parser.add_argument('--in-csv-2', type=str)
    parser.add_argument('--column-flag', type=str)
    args = parser.parse_args()

    logger.info(f'Reading {args.in_csv_1}')
    rvs1 = pd.read_csv(args.in_csv_1)[args.column_flag].to_numpy()
    logger.info(f'Data length {len(rvs1)}')

    logger.info(f'Reading {args.in_csv_2}')
    rvs2 = pd.read_csv(args.in_csv_2)[args.column_flag].to_numpy()
    logger.info(f'Data length {len(rvs2)}')

    ks_val = ttest_ind(rvs1, rvs2)
    logger.info(f'KS test: {ks_val}')


if __name__ == '__main__':
    main()