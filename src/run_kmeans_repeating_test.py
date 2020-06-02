import argparse
from tools.utils import get_logger
import pandas as pd
from tools.clustering import ClusterAnalysisDimAnalyzer
from tools.data_io import load_object


logger = get_logger('KMeans')


def main():
    parser = argparse.ArgumentParser(description='KMean clustering analysis')
    parser.add_argument('--in-data-dict-bin', type=str)
    parser.add_argument('--n-features', type=int)
    parser.add_argument('--out-png-folder', type=str)
    parser.add_argument('--n-cluster', type=int, default=10)
    parser.add_argument('--field-flag', type=str)
    parser.add_argument('--n-repeat', type=int)
    args = parser.parse_args()

    in_data_dict = load_object(args.in_data_dict_bin)
    kmean_analyzer = ClusterAnalysisDimAnalyzer(in_data_dict, args.n_features)

    kmean_analyzer.run_repeating_test(
        args.field_flag,
        args.n_cluster,
        args.n_repeat,
        args.out_png_folder)


if __name__ == '__main__':
    main()