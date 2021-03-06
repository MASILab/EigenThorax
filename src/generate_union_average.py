import argparse
from tools.config import load_config
from tools.average import AverageScans
from tools.data_io import DataFolder


def main():
    parser = argparse.ArgumentParser(description='ThoraxPCA')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Folder of input data', required=True)
    parser.add_argument('--save-path', type=str,
                        help='Output location for union average', required=True)
    parser.add_argument('--data-file-list', type=str,
                        help='Data file list')
    args = parser.parse_args()

    config = load_config(args.config)
    average_tool = None
    if args.data_file_list:
        data_folder_obj = DataFolder(args.in_folder, args.data_file_list)
        average_tool = AverageScans(config, in_data_folder_obj=data_folder_obj)
    else:
        average_tool = AverageScans(config, args.in_folder)

    average_tool.get_average_image_union(args.save_path)


if __name__ == '__main__':
    main()
