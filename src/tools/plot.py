import os
from tools.utils import get_logger
from tools.paral import AbstractParallelRoutine
from tools.data_io import DataFolder, ScanWrapper
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, exposure
from matplotlib import colors


logger = get_logger('Plot')


class OverlayMaskPNG(AbstractParallelRoutine):
    def __init__(self, config, in_folder, mask_img, out_png_folder, file_list_txt):
        super().__init__(config, in_folder, file_list_txt)
        self._mask_img = ScanWrapper(mask_img)
        self._out_png_folder = DataFolder.get_data_folder_obj(config, out_png_folder, data_list_txt=file_list_txt)
        self._out_png_folder.change_suffix('.png')
        self._vmax = 500
        self._vmin = -1000

    def _run_single_scan(self, idx):
        in_img_path = self._in_data_folder.get_file_path(idx)
        in_img = ScanWrapper(in_img_path)

        slice_in_img = self._clip_nifti(in_img.get_data())
        slice_mask_img = self._clip_nifti(self._mask_img.get_data())

        plt.figure(figsize=(15, 15))
        plt.axis('off')

        clip_x_nii_rescale = exposure.rescale_intensity(slice_in_img, in_range=(self._vmin, self._vmax), out_range=(0, 1))
        clip_x_nii_rgb = color.gray2rgb(clip_x_nii_rescale)
        plt.imshow(clip_x_nii_rgb, alpha=0.8)
        plt.imshow(slice_mask_img,
                   interpolation='none',
                   cmap='jet',
                   norm=colors.Normalize(vmin=0, vmax=1),
                   alpha=0.3)

        out_png_path = self._out_png_folder.get_file_path(idx)
        print(f'Saving image to {out_png_path}')
        plt.savefig(out_png_path, bbox_inches='tight', pad_inches=0)

    @staticmethod
    def _clip_nifti(im_data, offset=0):
        im_shape = im_data.shape
        clip_x = im_data[int(im_shape[0] / 2) - 1 + offset, :, :]
        clip_x = np.flip(clip_x, 0)
        clip_x = np.rot90(clip_x)

        return clip_x