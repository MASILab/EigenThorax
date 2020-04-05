import argparse
import numpy as np
import nibabel as nib


def apply_mask(in_img, mask_img, ambient_val, out_img):
    ori_img_obj = nib.load(in_img)
    mask_img_obj = nib.load(mask_img)

    ori_img = ori_img_obj.get_data()
    mask_img = mask_img_obj.get_data()

    masked_img = np.zeros(ori_img.shape)
    masked_img = np.add(masked_img, ambient_val)

    masked_img[mask_img > 0] = ori_img[mask_img > 0]

    masked_img_obj = nib.Nifti1Image(masked_img, affine=ori_img_obj.affine, header=ori_img_obj.header)
    nib.save(masked_img_obj, out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori', type=str)
    parser.add_argument('--mask', type=str)
    parser.add_argument('--ambient', type=float, default=-1000)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    apply_mask(args.ori, args.mask, args.ambient, args.out)

