from deeds_registration import *

if __name__ == '__main__':
    binary_dir = '/home-nfs2/local/VANDERBILT/xuk9/03-Projects/deedsBCV_yuankai'

    registrater = deeds_registration(binary_dir)

    target_img = '/home/local/VANDERBILT/xuk9/cur_data_dir/registration/nonrigid_deedsBCV/20200130_atlas_to_image/preprocess/moving1_resample.nii.gz'
    source_img = '/home/local/VANDERBILT/xuk9/cur_data_dir/registration/nonrigid_deedsBCV/20200130_atlas_to_image/atlas/labels.nii.gz'
    deformed_img = '/home/local/VANDERBILT/xuk9/cur_data_dir/registration/nonrigid_deedsBCV/20200130_atlas_to_image/reg/moving1_resample.nii.gz_deformed.nii.gz'
    aff_mtx_file = '/home/local/VANDERBILT/xuk9/cur_data_dir/registration/nonrigid_deedsBCV/20200130_atlas_to_image/omat/moving1_resample.nii.gz_matrix.txt'
    output = '/home/local/VANDERBILT/xuk9/cur_data_dir/registration/nonrigid_deedsBCV/20200130_atlas_to_image/interp/label_moving1_resample.nii.gz'

    registrater.deeds_apply_origin_seg(target_img, source_img, deformed_img, aff_mtx_file, output)