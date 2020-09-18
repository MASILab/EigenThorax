import argparse
from os import path
from tools.logit import GetLogitResultCrossValidation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-feature-matrix-bin', type=str)
    parser.add_argument('--num-fold', type=int)
    parser.add_argument('--num-pc', type=str)
    parser.add_argument('--out-folder', type=str)
    parser.add_argument('--scan-label-file', type=str)
    args = parser.parse_args()

    gaussian_logit_model_cv_obj = GetLogitResultCrossValidation(args.num_fold)
    gaussian_logit_model_cv_obj.load_data_single(args.in_feature_matrix_bin, args.num_pc)
    gaussian_logit_model_cv_obj.load_label_file(args.scan_label_file)
    gaussian_logit_model_cv_obj.create_cross_validation_folds()
    gaussian_logit_model_cv_obj.get_gaussian_fit_model_fold()
    gaussian_logit_model_cv_obj.get_logit_model_fold()
    gaussian_logit_model_cv_obj.get_validation_result()
    gaussian_logit_model_cv_obj.get_PLCOm2012_validation_statics()

    # out_obj_bin = path.join(args.out_folder, 'cv_data.bin')
    # gaussian_logit_model_cv_obj.save_validation_result_to_bin(out_obj_bin)

    # out_png_roc_auc = path.join(args.out_folder, 'roc_auc_linear.png')
    # gaussian_logit_model_cv_obj.plot_auc_roc_with_CI(out_png_roc_auc)
    #
    # out_png_order_compare = path.join(args.out_folder, 'roc_auc_order_compare.png')
    # gaussian_logit_model_cv_obj.plot_auc_roc_with_CI_4_order(out_png_order_compare)
    #
    # out_png_prc_auc = path.join(args.out_folder, 'prc_auc_linear.png')
    # gaussian_logit_model_cv_obj.plot_auc_prc_with_CI(out_png_prc_auc)
    #
    # out_png_order_compare = path.join(args.out_folder, 'prc_auc_order_compare.png')
    # gaussian_logit_model_cv_obj.plot_auc_prc_with_CI_4_order(out_png_order_compare)
    #
    # out_png_linear_roc_prc = path.join(args.out_folder, 'linear_roc_prc.png')
    # gaussian_logit_model_cv_obj.plot_roc_prc_linear_model(out_png_linear_roc_prc)

    # out_png_linear_quadratic_roc_prc = path.join(args.out_folder, 'linear_quad_roc_prc.png')
    # gaussian_logit_model_cv_obj.plot_linear_quadratic_roc_prc(out_png_linear_quadratic_roc_prc)

    out_png_compare_two_means = path.join(args.out_folder, 'compare_two_mean.png')
    gaussian_logit_model_cv_obj.plot_roc_auc_show_different_mean(out_png_compare_two_means)

    # PLCOm2012_validation_result = gaussian_logit_model_cv_obj.get_PLCOm2012_validation_statics()
    # PLCOm2012_roc_auc = PLCOm2012_validation_result['roc_auc']
    # PLCOm2012_prc_auc = PLCOm2012_validation_result['prc_auc']
    # print(f'auc_roc = {PLCOm2012_roc_auc}')
    # print(f'auc_prc = {PLCOm2012_prc_auc}')


if __name__ == '__main__':
    main()