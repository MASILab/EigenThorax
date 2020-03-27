import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from tools.data_io import DataFolder, ScanWrapper
import matplotlib.gridspec as gridspec
import os
from tools.utils import mkdir_p

show_pc_number = 10

def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--pc-folder', type=str,
                        help='Location of the principle feature images')
    parser.add_argument('--file-list-txt', type=str)
    # parser.add_argument('--out-png', type=str)
    parser.add_argument('--out-png-folder', type=str)
    parser.add_argument('--vmin', type=float, default=-0.001)
    parser.add_argument('--vmax', type=float, default=0.001)
    args = parser.parse_args()

    slices_x, slices_y, slices_z = _get_slices_dim_list(args.pc_folder, args.file_list_txt)

    mkdir_p(args.out_png_folder)
    out_png_x = os.path.join(args.out_png_folder, 'sagittal.png')
    out_png_y = os.path.join(args.out_png_folder, 'coronal.png')
    out_png_z = os.path.join(args.out_png_folder, 'axial.png')
    _draw_10_pc_plot(slices_x, out_png_x, args.vmax, args.vmin)
    _draw_10_pc_plot(slices_y, out_png_y, args.vmax, args.vmin)
    _draw_10_pc_plot(slices_z, out_png_z, args.vmax, args.vmin)


def _draw_10_pc_plot(slices_list, out_png, vmax, vmin):
    # plt.figure(figsize=(110, 50))
    plt.figure(figsize=(75, 30))
    gs1 = gridspec.GridSpec(2, 5)
    gs1.update(wspace=0.025, hspace=0.05)

    font = {'weight': 'bold',
            'size': 30}

    matplotlib.rc('font', **font)

    for file_idx in range(show_pc_number):
        slice_img = slices_list[file_idx]
        ax = plt.subplot(gs1[file_idx])
        plt.axis('off')
        plt.imshow(slice_img,
                   interpolation='none',
                   cmap='gray',
                   vmax=vmax,
                   vmin=vmin)
        ax.set_title(f'PC {file_idx + 1}', fontsize=60)

    print(f'Saving image to {out_png}', flush=True)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0)


def _get_slices_dim_list(pc_folder, file_list_txt):
    center_slices_x = []
    center_slices_y = []
    center_slices_z = []
    data_folder = DataFolder(pc_folder, file_list_txt)
    for file_idx in range(show_pc_number):
        data_folder.print_idx(file_idx)
        file_path = data_folder.get_file_path(file_idx)
        scan = ScanWrapper(file_path)
        slice_x, slice_y, slice_z = scan.get_center_slices()
        center_slices_x.append(slice_x)
        center_slices_y.append(slice_y)
        center_slices_z.append(slice_z)

    return center_slices_x, center_slices_y, center_slices_z



if __name__ == '__main__':
    main()