import argparse
from tools.config import load_config
from tools.preprocess import NonNanRegion


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--out-folder', type=str,
                        help='Output location for preprocessed images', required=True)
    parser.add_argument('--file-list-txt', type=str,
                        help='Data file list')
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess_obj = NonNanRegion(
        config,
        args.in_folder,
        args.out_folder,
        args.file_list_txt
    )
    preprocess_obj.run_parallel()


if __name__ == '__main__':
    main()
