import nibabel as nib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def main():
    parser = argparse.ArgumentParser(
        description='Plot the tri-view in color and plot the color bar')
    parser.add_argument('--in-img', type=str)
    parser.add_argument('--out-png', type=str)
    parser.add_argument('--vmin', type=float, default=-2)
    parser.add_argument('--vmax', type=float, default=2)
    args = parser.parse_args()

    clips = clip_nifti(args.in_img)

    ori_out_png = args.out_png

    clips_data = {
        's': {
            'out_png': ori_out_png.replace('.png', '_s.png'),
            'slice': clips[0],
            'title': 'Sagittal'
        },
        'c': {
            'out_png': ori_out_png.replace('.png', '_c.png'),
            'slice': clips[1],
            'title': 'Coronal'
        },
        'a': {
            'out_png': ori_out_png.replace('.png', '_a.png'),
            'slice': clips[2],
            'title': 'Axial'
        }
    }

    for key in clips_data:
        if_color_bar = True
        # if key is 's':
        #     if_color_bar = True
        title_str = clips_data[key]['title']
        get_slice_png(clips_data[key]['slice'],
                      clips_data[key]['out_png'],
                      args.vmax, args.vmin,
                      if_color_bar, title_str)


def get_slice_png(slice_data, out_png, vmax, vmin, if_color_bar, title_str):

    fig, axs = plt.subplots(1, 1)

    img = axs.imshow(
        slice_data,
        cmap='jet',
        vmax=vmax,
        vmin=vmin,
        interpolation='bicubic'
    )
    plt.axis('off')
    axs.set_title(title_str)

    if if_color_bar:
        axins = inset_axes(axs,
                           width="5%",
                           height="100%",
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axs.transAxes,
                           borderpad=0)
        cbar = fig.colorbar(img, cax=axins)
        cbar.minorticks_on()

    print(f'Saving image to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0.2)



def clip_nifti(nii_file_path):
    print('Clippnig image %s' % nii_file_path)
    im = nib.load(nii_file_path)
    im_data = im.get_fdata()
    im_shape = im_data.shape
    clip_x = im_data[int(im_shape[0]/2)-1 , :, :].transpose()
    clip_x = np.flip(clip_x, 0)
    clip_x = np.flip(clip_x, 1)
    clip_y = im_data[:, int(im_shape[1]/2)-1 + 30, :].transpose()
    clip_y = np.flip(clip_y, 0)
    clip_z = im_data[:, :, int(im_shape[2]/2)-1].transpose()
    clip_z = np.flip(clip_z, 0)

    clips = []
    clips.append(clip_x)
    clips.append(clip_y)
    clips.append(clip_z)

    return clips


if __name__ == '__main__':
    main()

