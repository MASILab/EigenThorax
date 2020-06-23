import argparse
from tools.data_io import load_object
from tools.utils import get_logger
from tools.data_io import ClusterAnalysisDataDict
from tools.clustering import ClusterAnalysisSearchNumCluster
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


logger = get_logger('Num cluster')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--bin-folder', type=str)
    parser.add_argument('--out-png-folder', type=str)
    args = parser.parse_args()

    bin_data_dict_path_list = []
    bin_data_dict_name_list = []
    bin_data_dict_n_feature = []
    bin_data_dict_path_list.append(os.path.join(args.bin_folder, 'init_data_dict.bin'))
    bin_data_dict_name_list.append('original (#dim=20)')
    bin_data_dict_n_feature.append(20)
    # bin_data_dict_path_list.append(os.path.join(args.bin_folder, 'reduct_bmi_data_dict.bin'))
    # bin_data_dict_name_list.append('reduce BMI (#dim=19)')
    # bin_data_dict_n_feature.append(19)
    # bin_data_dict_path_list.append(os.path.join(args.bin_folder, 'reduct_bmi_2_data_dict.bin'))
    # bin_data_dict_name_list.append('reduce BMI (#dim=18)')
    # bin_data_dict_n_feature.append(18)
    # bin_data_dict_path_list.append(os.path.join(args.bin_folder, 'reduct_age_1_data_dict.bin'))
    # bin_data_dict_name_list.append('reduce Age (#dim=17)')
    # bin_data_dict_n_feature.append(16)
    # bin_data_dict_path_list.append(os.path.join(args.bin_folder, 'reduct_packyear_1_data_dict.bin'))
    # bin_data_dict_name_list.append('reduce Packyear (#dim=16)')
    # bin_data_dict_n_feature.append(15)
    bin_data_dict_path_list.append(os.path.join(args.bin_folder, 'reduct_packyear_1_data_dict.bin'))
    bin_data_dict_name_list.append('reduce BMI, Age and Packyear (#dim=15)')
    bin_data_dict_n_feature.append(15)

    num_bin_data = 2

    n_cluster_range = range(1, 11)

    fig, ax = plt.subplots(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2)

    ax_list = []
    for idx_ax in range(4):
        ax_list.append(plt.subplot(gs[idx_ax]))

    idx_ax = 0
    for idx_bin_data in range(num_bin_data):
        bin_data_dict = load_object(bin_data_dict_path_list[idx_bin_data])
        bin_data_name = bin_data_dict_name_list[idx_bin_data]
        bin_data_num_features = bin_data_dict_n_feature[idx_bin_data]
        data_dict_obj = ClusterAnalysisDataDict(bin_data_dict, bin_data_num_features)
        optimal_cluster_num_obj = ClusterAnalysisSearchNumCluster(data_dict_obj)

        elbow_list, silhouette_list = optimal_cluster_num_obj.get_elbow_and_silhouette_array()

        ax_list[idx_ax].plot(n_cluster_range,
                 elbow_list,
                 label=bin_data_name)
        ax_list[idx_ax].set_title('Sum of squared distance to cluster centroids')
        idx_ax += 1

        ax_list[idx_ax].plot(n_cluster_range[1:],
                 silhouette_list[1:],
                 label=bin_data_name)
        ax_list[idx_ax].set_title('Silhouette score')
        idx_ax += 1

    for idx_ax in range(4):
        ax_list[idx_ax].legend(loc='best')

    out_png = os.path.join(args.out_png_folder, 'optimal_num_cluster.png')
    logger.info(f'Save to {out_png}')
    fig.tight_layout()
    plt.savefig(out_png)
    plt.close()


if __name__ == '__main__':
    main()