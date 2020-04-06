import argparse
from tools.config import load_config
from tools.preprocess import MaskIntersection


def main():
    parser = argparse.ArgumentParser(description='Run preprocess on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Folder of input mask data', required=True)
    parser.add_argument('--out-mask', type=str,
                        help='Output path of mask data')
    parser.add_argument('--file-list-txt', type=str,
                        help='Data file list')
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess_obj = MaskIntersection(
        config,
        args.in_folder,
        args.out_mask,
        args.file_list_txt
    )
    preprocess_obj.run()


if __name__ == '__main__':
    main()
