import argparse
from tools.pca import PCA_NII_3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from tools.data_io import DataFolder, ScanWrapper
import matplotlib.gridspec as gridspec
import os
from tools.utils import get_logger
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


logger = get_logger('Plot - PC')


class PlotPCGrid:
    def __init__(self,
                 pc_folder_obj,
                 step_axial,
                 step_sagittal,
                 step_coronal,
                 num_show_pc):
        self._pc_folder_obj = pc_folder_obj
        self._step_axial = step_axial,
        self._step_coronal = step_coronal,
        self._step_sagittal = step_sagittal,
        self._num_show_pc = num_show_pc
        self._cm = 'hsv'
        self._num_view = 3
        self._out_dpi = 20
        self._num_clip = 1
        self._sub_title_font_size = 60

    def plot_pc(self, out_png):
        fig = plt.figure(figsize=(self._num_show_pc * 20, self._num_clip * self._num_view * 15))
        gs = gridspec.GridSpec(1, self._num_show_pc)

        sub_gs_list = []
        for idx_pc in range(self._num_show_pc):
            sub_gs = gs[idx_pc].subgridspec(self._num_clip * self._num_view, 1)
            sub_gs_list.append(sub_gs)

        for idx_pc in range(self._num_show_pc):
            img_data_obj = ScanWrapper(self._pc_folder_obj.get_file_path(idx_pc))
            img_data = img_data_obj.get_data()
            img_name = os.path.basename(self._pc_folder_obj.get_file_path(idx_pc)).replace('.nii.gz', '')
            logger.info(f'Reading image {self._pc_folder_obj.get_file_path(idx_pc)}')
            self._plot_one_pc(img_data, sub_gs_list[idx_pc], f'{img_name}')

        # out_eps = out_png.replace('.png', '.eps')
        logger.info(f'Save fig to {out_png}')
        # plt.savefig(out_eps, bbox_inches='tight', pad_inches=0, dpi=self._out_dpi, format='pdf')
        plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=self._out_dpi)
        plt.close(fig=fig)

    def _plot_one_pc(self, image_data, sub_gs, title_prefix):
        # vmin = np.min(image_data)
        # vmax = np.max(image_data)
        range_norm = np.max([np.min(image_data), np.max(image_data)])
        vmin = - 0.5 * range_norm
        vmax = 0.5 * range_norm

        # vmin = -8
        # vmax = 1

        view_config_list = self._get_view_config()
        for idx_view in range(self._num_view):
            clip_plane = view_config_list[idx_view]['clip plane']
            step_size = view_config_list[idx_view]['step size'][0]
            for idx_clip in range(self._num_clip):
                # logger.info(f'Plot view ({idx_clip, idx_view})')
                # clip_off_set = (idx_clip - 2) * step_size
                clip_off_set = 0
                clip = self._clip_image(image_data, clip_plane, clip_off_set)
                ax = plt.subplot(sub_gs[idx_clip + idx_view * self._num_clip, 0])
                plt.axis('off')
                im = plt.imshow(
                    clip,
                    interpolation='none',
                    cmap=self._cm,
                    norm=colors.Normalize(vmin=vmin, vmax=vmax)
                )
                ax.set_title(f'{title_prefix}_{clip_plane}', fontsize=self._sub_title_font_size)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                cb = plt.colorbar(im, cax=cax)
                # cb.set_label('Intensity of Eigen Image')
                cb.ax.tick_params(labelsize=self._sub_title_font_size/2)

    def _get_view_config(self):
        view_config_list = [
            {
                'clip plane': 'Axial',
                'step size': self._step_axial
            },
            {
                'clip plane': 'Sagittal',
                'step size': self._step_sagittal
            },
            {
                'clip plane': 'Coronal',
                'step size': self._step_coronal
            }
        ]
        return view_config_list

    @staticmethod
    def _clip_image(image_data, clip_plane, offset=0):
        im_shape = image_data.shape
        clip = None
        if clip_plane == 'Sagittal':
            clip = image_data[int(im_shape[0] / 2) - 1 + offset, :, :]
            clip = np.flip(clip, 0)
            clip = np.rot90(clip)
        elif clip_plane == 'Coronal':
            clip = image_data[:, int(im_shape[1] / 2) - 1 + offset, :]
            clip = np.rot90(clip)
        elif clip_plane == 'Axial':
            clip = image_data[:, :, int(im_shape[2] / 2) - 1 + offset]
            clip = np.rot90(clip)
        else:
            raise NotImplementedError

        return clip


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--pc-folder', type=str,
                        help='Location of the principle feature images')
    parser.add_argument('--file-list-txt', type=str)
    parser.add_argument('--out-png', type=str)
    parser.add_argument('--step-axial', type=int, default=50)
    parser.add_argument('--step-sagittal', type=int, default=75)
    parser.add_argument('--step-coronal', type=int, default=30)
    parser.add_argument('--num-show-pc', type=int, default=10)
    args = parser.parse_args()

    pc_folder_obj = DataFolder(args.pc_folder, args.file_list_txt)
    plt_obj = PlotPCGrid(
        pc_folder_obj,
        args.step_axial,
        args.step_sagittal,
        args.step_coronal,
        args.num_show_pc
    )
    plt_obj.plot_pc(args.out_png)


if __name__ == '__main__':
    main()