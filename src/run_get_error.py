import argparse
from tools.config import load_config
from tools.loss import GetLossBetweenFolder
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Run resample on in data folder')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder-1', type=str,
                        help='Folder 1 of input data', required=True)
    parser.add_argument('--in-folder-2', type=str,
                        help='Folder 2 of input data', required=True)
    parser.add_argument('--file-list-txt', type=str,
                        help='Help to define the order of the files')
    parser.add_argument('--out-png-path', type=str,
                        help='Output as png')
    args = parser.parse_args()

    config = load_config(args.config)
    loss = GetLossBetweenFolder(config, args.in_folder_1, args.in_folder_2, args.file_list_txt)
    loss.print_file_list()
    loss.run_non_parallel()

    array_nrmse = loss.get_nrmse()
    print(array_nrmse, flush=True)

    plt.figure(figsize=(30, 15))

    x = range(20)
    plt.plot(x, array_nrmse, 'ro-', linewidth=2)
    plt.xlabel('PCs')
    plt.title('nrmse diff')

    print(f'Plot nrmse to {args.out_png_path}')
    plt.savefig(args.out_png_path, bbox_inches='tight', pad_inches=0)



if __name__ == '__main__':
    main()