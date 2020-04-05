import argparse
from tools.utils import mkdir_p
from tools.plot import OverlayMaskPNG
from tools.config import load_config


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--config', type=str,
                        help='Path to the YAML config file', required=True)
    parser.add_argument('--in-folder', type=str,
                        help='Location of the input images')
    parser.add_argument('--mask-img', type=str,
                        help='Binary mask image')
    parser.add_argument('--out-png-folder', type=str,
                        help='Output png file folder')
    parser.add_argument('--file-list-txt', type=str)
    args = parser.parse_args()

    mkdir_p(args.out_png_folder)
    config = load_config(args.config)
    plot_obj = OverlayMaskPNG(
        config,
        args.in_folder,
        args.mask_img,
        args.out_png_folder,
        args.file_list_txt
    )
    plot_obj.run_parallel()


if __name__ == '__main__':
    main()