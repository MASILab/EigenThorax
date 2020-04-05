import argparse
from tools.config import load_config
from tools.preprocess import CalcJacobian


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--out-folder', type=str,
                        help='Output location for preprocessed images', required=True)
    parser.add_argument('--ref-img', type=str,
                        help='Use this average image to impute the missing voxels in targets')
    parser.add_argument('--data-file-list', type=str,
                        help='Data file list')
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess_obj = CalcJacobian(
        config,
        args.in_folder,
        args.out_folder,
        args.ref_img,
        args.data_file_list
    )
    preprocess_obj.run_parallel()


if __name__ == '__main__':
    main()