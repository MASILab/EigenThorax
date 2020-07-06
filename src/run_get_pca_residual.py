import argparse
from tools.pca_analysis import ParalGetResidual
from tools.data_io import ScanWrapper, DataFolder, save_object
from tools.utils import get_logger


logger = get_logger('PCA get residual')


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--load-pca-bin-path', type=str,
                        help='Location of the pca bin')
    parser.add_argument('--in-folder', type=str)
    parser.add_argument('--out-folder', type=str)
    parser.add_argument('--mask-img', type=str)
    parser.add_argument('--num-mode', type=int)
    parser.add_argument('--file-list-txt', type=str)
    parser.add_argument('--num-process', type=int, default=10)
    args = parser.parse_args()

    in_folder_obj = DataFolder(args.in_folder, args.file_list_txt)
    out_folder_obj = DataFolder(args.out_folder, args.file_list_txt)
    mask_img_obj = ScanWrapper(args.mask_img)

    paral_obj = ParalGetResidual(
        args.load_pca_bin_path,
        in_folder_obj,
        out_folder_obj,
        mask_img_obj,
        args.num_mode,
        args.num_process
    )

    paral_obj.run_parallel()


if __name__ == '__main__':
    main()