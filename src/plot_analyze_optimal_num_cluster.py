import argparse
from tools.data_io import load_object
from tools.utils import get_logger
from tools.data_io import ClusterAnalysisDataDict
from tools.clustering import ClusterAnalysisSearchNumCluster
import os


logger = get_logger('LDA')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--in-data-dict-bin', type=str)
    parser.add_argument('--n-features', type=int)
    parser.add_argument('--out-png-folder', type=str)
    args = parser.parse_args()

    in_data_dict = load_object(args.in_data_dict_bin)
    data_dict_obj = ClusterAnalysisDataDict(in_data_dict, args.n_features)

    optimal_cluster_num_obj = ClusterAnalysisSearchNumCluster(data_dict_obj)
    out_elbow_png = os.path.join(args.out_png_folder, 'elbow_plot.png')
    out_silhouette_png = os.path.join(args.out_png_folder, 'silhouette_plot.png')
    optimal_cluster_num_obj.ElbowSilhouettePlot(out_elbow_png, out_silhouette_png)


if __name__ == '__main__':
    main()