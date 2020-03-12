import nibabel as nib
import argparse
import os
import numpy as np
from utils import get_extension, get_chunks_list
from multiprocessing import Pool

def main():
    parser = argparse.ArgumentParser(description='Generate average atlas for an image folder.')
    parser.add_argument('--in_folder', type=str,
                        help='The input image folder')
    parser.add_argument('--out_union', type=str,
                        help='The output image path (with .nii.gz)')
    parser.add_argument('--out_inter', type=str,
                        help='The output image path (with .nii.gz)', default='')
    parser.add_argument('--ref', type=str,
                        help='Path of reference image. Define the affine and header of output nii.gz')
    parser.add_argument('--num_processes', type=int, default=10)

    args = parser.parse_args()
    file_list_all = os.listdir(args.in_folder)
    print('Process images under folder: ', args.in_folder)
    print('Number of files in folder %s is %d' % (args.in_folder, len(file_list_all)))
    nifti_file_list = [file_path for file_path in file_list_all if get_extension(file_path) == '.gz']
    print('Number of nii.gz files: ', len(nifti_file_list))

    file_name_chunks = get_chunks_list(nifti_file_list, args.num_processes)

    pool = Pool(processes=args.num_processes)

    # Get the shape.
    # im_temp = nib.load(os.path.join(args.in_folder, nifti_file_list[0]))
    im_temp = nib.load(args.ref)
    im_header = im_temp.header
    im_affine = im_temp.affine
    im_temp_data = im_temp.get_data()
    im_shape = im_temp_data.shape

    averaged_image_union = np.zeros(im_shape)
    averaged_image_inter = np.zeros(im_shape)
    averaged_image_union.fill(np.nan)
    # averaged_image_inter.fill(np.nan)
    non_null_mask_count_image = np.zeros(im_shape)

    if args.out_inter != '':
        print('Average in intersection:')
        image_average_inter_result_list = [pool.apply_async(sum_images_inter,
                                                            (file_name_chunk, args.in_folder))
                                           for file_name_chunk in file_name_chunks]

        for thread_idx in range(len(image_average_inter_result_list)):
            result = image_average_inter_result_list[thread_idx]
            result.wait()
            print(f'Thread with idx {thread_idx} / {len(image_average_inter_result_list)} is completed')
            print('Adding to averaged_image...')
            averaged_image_chunk = result.get()
            averaged_image_inter = add_image_inter(averaged_image_inter, averaged_image_chunk)
            print('Done.')

        averaged_image_inter = np.divide(averaged_image_inter,
                                         len(nifti_file_list),
                                         out=averaged_image_inter,
                                         where=np.logical_not(np.isnan(averaged_image_inter)))
        average_image_inter_obj = nib.Nifti1Image(averaged_image_inter, affine=im_affine, header=im_header)
        print(f'Saving to {args.out_inter}')
        nib.save(average_image_inter_obj, args.out_inter)
        print('Done.')
        print('')

    print('Average in union')
    image_average_union_result_list = [pool.apply_async(sum_images_union,
                                                        (file_name_chunk, args.in_folder))
                                       for file_name_chunk in file_name_chunks]

    for thread_idx in range(len(image_average_union_result_list)):
        result = image_average_union_result_list[thread_idx]
        result.wait()
        print(f'Thread with idx {thread_idx} / {len(image_average_union_result_list)} is completed')
        print('Adding to averaged_image...')
        averaged_image_chunk = result.get()
        averaged_image_union = add_image_union(averaged_image_union, averaged_image_chunk)
        print('Done.')

    non_null_mask_count_result = [pool.apply_async(sum_non_null_count,
                                                   (file_name_chunk, args.in_folder))
                                  for file_name_chunk in file_name_chunks]

    for thread_idx in range(len(non_null_mask_count_result)):
        result = non_null_mask_count_result[thread_idx]
        result.wait()
        print(f'Thread with idx {thread_idx} / {len(non_null_mask_count_result)} is completed')
        print('Adding to averaged_image...')
        averaged_image_chunk = result.get()
        non_null_mask_count_image = np.add(non_null_mask_count_image, averaged_image_chunk)
        print('Done.')

    averaged_image_union = np.divide(averaged_image_union,
                                     non_null_mask_count_image,
                                     out=averaged_image_union,
                                     where=non_null_mask_count_image>0)

    averaged_image_union_obj = nib.Nifti1Image(averaged_image_union, affine=im_affine, header=im_header)
    nib.save(averaged_image_union_obj, args.out_union)
    print('Done.')


def sum_images_inter(file_list, in_folder):
    print('Sum images, intersect non-null region. Loading images...')
    im_temp = nib.load(os.path.join(in_folder, file_list[0]))
    im_temp_data = im_temp.get_data()

    sum_image = np.zeros_like(im_temp_data)

    for id_file in range(len(file_list)):
        file_name = file_list[id_file]
        print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
        file_path = os.path.join(in_folder, file_name)
        im = nib.load(file_path)
        im_data = im.get_data()
        sum_image = add_image_inter(sum_image, im_data)

    return sum_image


def sum_images_union(file_list, in_folder):
    print('Sum images, union non-null region. Loading images...')
    im_temp = nib.load(os.path.join(in_folder, file_list[0]))
    im_temp_data = im_temp.get_data()

    sum_image = np.full_like(im_temp_data, np.nan)

    for id_file in range(len(file_list)):
        file_name = file_list[id_file]
        print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
        file_path = os.path.join(in_folder, file_name)
        im = nib.load(file_path)
        im_data = im.get_data()
        sum_image = add_image_union(sum_image, im_data)

    return sum_image


def sum_non_null_count(file_list, in_folder):
    print('Count non-null per voxel. Loading images...')
    im_temp = nib.load(os.path.join(in_folder, file_list[0]))
    im_temp_data = im_temp.get_data()

    sum_image = np.zeros_like(im_temp_data)

    for id_file in range(len(file_list)):
        file_name = file_list[id_file]
        print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
        file_path = os.path.join(in_folder, file_name)
        im = nib.load(file_path)
        im_data = im.get_data()

        sum_image = np.add(sum_image, 1, out=sum_image, where=np.logical_not(np.isnan(im_data)))

    return sum_image


def add_image_inter(image1, image2):
    return np.add(image1, image2,
                  out=np.full_like(image1, np.nan),
                  where=np.logical_not(np.logical_or(np.isnan(image1),
                                                     np.isnan(image2))))


def add_image_union(image1, image2):
    add_image = np.full_like(image1, np.nan)
    add_image[np.logical_not(np.logical_and(np.isnan(image1),
                                            np.isnan(image2)))] = 0

    add_image = np.add(add_image, image1, out=add_image, where=np.logical_not(np.isnan(image1)))
    add_image = np.add(add_image, image2, out=add_image, where=np.logical_not(np.isnan(image2)))

    return add_image


def sum_images(file_list, in_folder):
    print('Sum images. Loading images...')
    im_temp = nib.load(os.path.join(in_folder, file_list[0]))
    im_temp_data = im_temp.get_data()
    im_shape = im_temp_data.shape

    sum_image = np.zeros(im_shape)

    for id_file in range(len(file_list)):
        file_name = file_list[id_file]
        print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
        file_path = os.path.join(in_folder, file_name)
        im = nib.load(file_path)
        im_data = im.get_data()
        sum_image = np.add(sum_image, im_data)

    return sum_image


def sum_images_mask(file_list, in_folder):
    print('Sum images. Loading images...')
    im_temp = nib.load(os.path.join(in_folder, file_list[0]))
    im_temp_data = im_temp.get_data()
    im_shape = im_temp_data.shape

    sum_image = np.zeros(im_shape)

    for id_file in range(len(file_list)):
        file_name = file_list[id_file]
        print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
        file_path = os.path.join(in_folder, file_name)
        im = nib.load(file_path)
        im_data = im.get_data()
        im_data[im_data>0] = 1
        im_data[im_data<0] = 0
        sum_image = np.add(sum_image, im_data)

    return sum_image


def sum_images_with_mask(file_list, in_folder, in_mask_folder):
    print('Sum images with masks. Loading images...')
    im_temp = nib.load(os.path.join(in_folder, file_list[0]))
    im_temp_data = im_temp.get_data()
    im_shape = im_temp_data.shape

    sum_image = np.zeros(im_shape)

    for id_file in range(len(file_list)):
        file_name = file_list[id_file]
        print('%s (%d/%d)' % (file_name, id_file, len(file_list)))
        image_file_path = os.path.join(in_folder, file_name)
        mask_file_path = os.path.join(in_mask_folder, file_name)
        image_obj = nib.load(image_file_path)
        mask_obj = nib.load(mask_file_path)
        image_data = image_obj.get_data()
        mask_data = mask_obj.get_data()
        mask_data[mask_data>0] = 1
        mask_data[mask_data<0] = 0
        sum_image = np.add(sum_image, image_data, out=sum_image, where=mask_data!=0)

    return sum_image


def load_scan(path):
    img=nib.load(path)
    return img #a nib object


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate average atlas for an image folder.')
    parser.add_argument('--in_folder', type=str,
                        help='The input image folder')
    parser.add_argument('--out_union', type=str,
                        help='The output image path (with .nii.gz)')
    parser.add_argument('--out_inter', type=str,
                        help='The output image path (with .nii.gz)', default='')
    parser.add_argument('--ref', type=str,
                        help='Path of reference image. Define the affine and header of output nii.gz')
    parser.add_argument('--num_processes', type=int, default=10)

    main()