import argparse
from os import path
from tools.logit import GetLogitResultCrossValidation
from tools.data_io import save_object


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

    # out_obj_bin = path.join(args.out_folder, 'cv_data.bin')
    # gaussian_logit_model_cv_obj.save_validation_result_to_bin(out_obj_bin)

    out_png_roc_auc = path.join(args.out_folder, 'roc_auc.png')
    gaussian_logit_model_cv_obj.plot_auc_roc_with_CI(out_png_roc_auc)


if __name__ == '__main__':
    main()