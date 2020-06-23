import argparse
from tools.clinical import ClinicalDataReaderSPORE
from tools.data_io import load_object
from tools.utils import get_logger
from tools.data_io import ClusterAnalysisDataDict
from tools.correlation_analysis import CorrelationAnalysis, CorrelationAnalysis2OrthoSpace
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

    # corr_analysis_obj = CorrelationAnalysis(data_dict_obj)
    # max_2_bmi = corr_analysis_obj.correlation_bar_plot('bmi', args.out_png_folder)
    # max_2_age = corr_analysis_obj.correlation_bar_plot('Age', args.out_png_folder)
    # max_2_packyear = corr_analysis_obj.correlation_bar_plot('Packyear', args.out_png_folder)
    #
    # corr_analysis_obj.mutual_info_bar_plot('bmi', args.out_png_folder)
    # corr_analysis_obj.mutual_info_bar_plot('Age', args.out_png_folder)
    # corr_analysis_obj.mutual_info_bar_plot('Packyear', args.out_png_folder)

    # corr_analysis_obj.plot_2D_dim_plot(max_2_bmi, 'bmi', args.out_png_folder)
    # corr_analysis_obj.plot_2D_dim_plot(max_2_age, 'Age', args.out_png_folder)
    # corr_analysis_obj.plot_2D_dim_plot(max_2_packyear, 'Packyear', args.out_png_folder)

    corr_analysis_ortho_obj = CorrelationAnalysis2OrthoSpace(data_dict_obj)

    # corr_analysis_ortho_obj.plot_2D_top_dim_ortho('bmi', args.out_png_folder)
    # corr_analysis_ortho_obj.plot_2D_top_dim_ortho('Age', args.out_png_folder)
    # corr_analysis_ortho_obj.plot_2D_top_dim_ortho('Packyear', args.out_png_folder)
    # corr_analysis_ortho_obj.plot_2D_top_dim_lda_ortho('CAC', args.out_png_folder)
    # corr_analysis_ortho_obj.plot_2D_top_dim_lda_ortho('COPD', args.out_png_folder)
    # corr_analysis_ortho_obj.plot_2D_top_dim_lda_ortho('CancerSubjectFirstScan', args.out_png_folder)
    # corr_analysis_ortho_obj.plot_2D_top_dim_ortho('CancerSubjectFirstScan', args.out_png_folder)

    # corr_analysis_ortho_obj.plot_2D_grid_pack_field_list(args.out_png_folder)
    corr_analysis_ortho_obj.plot_2D_grid_pack_field_tsne_list(args.out_png_folder)


if __name__ == '__main__':
    main()