import argparse
from tools.config import load_config
from tools.average import AverageValidRegion


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-scan-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--in-mask-folder', type=str,
                        help='Folder of valid region mask')
    parser.add_argument('--out-average-img', type=str,
                        help='Output location of averaged image')
    parser.add_argument('--file-list-txt', type=str,
                        help='Data file list')
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess_obj = AverageValidRegion(
        config,
        args.in_scan_folder,
        args.in_mask_folder,
        args.file_list_txt,
        args.out_average_img
    )
    preprocess_obj.run_get_average()


if __name__ == '__main__':
    main()
