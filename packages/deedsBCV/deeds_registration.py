import os
import nibabel as nib
import numpy as np
from skimage.transform import resize

class deeds_registration(object):
    """a class to perform deeds registration
       """

    def __init__(self, binary_dir, iso_radius=2, padding=True):
        self.binary_dir = binary_dir
        self.iso_radius = iso_radius
        self.padding = padding

    def c3d_resample(self, input_img, output_img, padding=False):
        """Return the balance remaining after withdrawing *amount*
        dollars."""
        if not os.path.exists(output_img):
            resample_para = '-resample-mm %dx%dx%dmm' % (self.iso_radius, self.iso_radius, self.iso_radius)
            # origin_voxel_para = '-origin-voxel 0x0x0vox'
            origin_voxel_para = ''
            padding_para = '-pad 0x0x14vox 0x0x14vox -1024'

            if padding:
                region_para = '-region 0x0x20vox 180x140x190vox'
                cmd = '%s/c3d %s %s %s %s -o %s' % (self.binary_dir, input_img, resample_para, origin_voxel_para, padding_para, output_img)
            else:
                region_para = '-region 0x0x10vox 180x140x190vox'
                cmd = '%s/c3d %s %s %s -o %s' % (self.binary_dir, input_img, resample_para, origin_voxel_para, output_img)
            os.system(cmd)

    def c3d_padding(self, input_img, output_img):
        padding_para = '-pad 14x0x0vox 14x0x0vox -1024'
        cmd = '%s/c3d %s %s -o %s' % (self.binary_dir, input_img, padding_para, output_img)
        os.system(cmd)

    def affine_reg(self, target_img, source_img, aff_mtx):
        output_file = aff_mtx + '_matrix.txt'
        if not os.path.exists(output_file):
            source_para = '-M %s' % source_img
            target_para = '-F %s' % target_img
            output_para = '-O %s' % aff_mtx
            cmd = '%s/linearBCV %s %s %s' % (self.binary_dir, target_para, source_para, output_para)
            os.system(cmd)
        return output_file


    def deeds_reg(self, target_img, source_img, aff_mtx_file, output):
        output_file = output + '_deformed.nii.gz'
        if not os.path.exists(output_file):
            source_para = '-M %s' % source_img
            target_para = '-F %s' % target_img
            output_para = '-O %s' % output
            affine_mtx_para = '-A %s' % aff_mtx_file
            cmd = '%s/deedsBCVwinv %s %s %s %s' % (self.binary_dir, target_para, source_para, output_para, affine_mtx_para)
            os.system(cmd)
        return output_file


    def deeds_apply_origin(self, target_img, source_img, deformed_img, aff_mtx_file, output):
        output_file = output
        if not os.path.exists(output_file):
            source_para = '-M %s' % source_img
            deformed_para = '-D %s' % output
            target_para = '-F %s' % target_img
            output_para = '-O %s' % deformed_img
            affine_mtx_para = '-A %s' % aff_mtx_file
            cmd = '%s/applyBCV_orires %s %s %s %s %s' % (self.binary_dir, target_para, source_para, deformed_para, output_para, affine_mtx_para)
            os.system(cmd)
        return output_file

    def deeds_apply_origin_seg(self, target_img, source_img, deformed_img, aff_mtx_file, output):
        output_file = output
        if not os.path.exists(output_file):
            source_para = '-M %s' % source_img
            deformed_para = '-D %s' % output
            target_para = '-F %s' % target_img
            output_para = '-O %s' % deformed_img
            affine_mtx_para = '-A %s' % aff_mtx_file
            cmd = '%s/applyBCV_orires_Seg %s %s %s %s %s' % (self.binary_dir, target_para, source_para, deformed_para, output_para, affine_mtx_para)
            print(cmd)
            # os.system(cmd)
        return output_file

    def deeds_apply(self, target_img, source_img, deformed_img, aff_mtx_file, output):
        output_file = output
        if not os.path.exists(output_file):
            source_para = '-M %s' % source_img
            deformed_para = '-D %s' % output
            output_para = '-O %s' % deformed_img
            affine_mtx_para = '-A %s' % aff_mtx_file
            cmd = '%s/applyBCV %s %s %s' % (self.binary_dir, source_para, deformed_para, output_para, affine_mtx_para)
            os.system(cmd)
        return output_file


    def band_filter(self,input_img,output_img):
        if not os.path.exists(output_img):
            img_3d = nib.load(input_img)
            vol = img_3d.get_data()
            np.asarray(vol)
            vol[vol < -150] = -150
            vol[vol > 350] = 350
            # seg_img = nib.Nifti1Image(vol-resize,img_3d.affine,img_3d.header)
            out_vol = nib.Nifti1Image(vol, affine=img_3d.affine)
            nib.save(out_vol, output_img)

    def deeds_pipeline(self,target_img, source_img, output_img, temp_dir):
        if not os.path.exists(output_img):
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # get base name
            source_base_name = os.path.basename(source_img)
            source_base_name = source_base_name.replace('.nii.gz', '')
            target_base_name = os.path.basename(target_img)
            target_base_name = target_base_name.replace('.nii.gz', '')

            # run affine
            aff_mtx = os.path.join(temp_dir, source_base_name + '_affine')
            aff_mtx_file = self.affine_reg(target_img, source_img, aff_mtx)

            # run non-rigid
            output_file = os.path.join(temp_dir, source_base_name + '_affine_deeds')
            deeds_deformed_file = self.deeds_reg(target_img, source_img, aff_mtx_file, output_file)

            cp_cmd = 'cp %s %s' % (deeds_deformed_file,output_img)
            os.system(cp_cmd)
        return output_img

    def deeds_iso_pipeline(self, target_img, source_img, output_img, temp_dir):
        if not (os.path.exists(target_img) and os.path.exists(source_img)):
            return output_img

        if not os.path.exists(output_img):
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # get base name
            source_base_name = os.path.basename(source_img)
            source_base_name = source_base_name.replace('.nii.gz', '')
            target_base_name = os.path.basename(target_img)
            target_base_name = target_base_name.replace('.nii.gz', '')

            # #iso resample
            source_iso_img = os.path.join(temp_dir, source_base_name + '_res.nii.gz')
            target_iso_img = os.path.join(temp_dir, target_base_name + '_res.nii.gz')
            self.c3d_resample(source_img, source_iso_img)
            self.c3d_resample(target_img, target_iso_img)

            # # iso padding
            # target_padding_img = os.path.join(temp_dir, target_base_name + '_pad.nii.gz')
            # deeds.c3d_padding(target_iso_img, target_padding_img)

            # run affine
            aff_mtx = os.path.join(temp_dir, source_base_name + '_affine')
            aff_mtx_file = self.affine_reg(target_iso_img, source_iso_img, aff_mtx)

            # see if affine is nan
            text_file = open(aff_mtx_file, 'r')
            lines = text_file.readlines()
            if lines[0].find('nan') > 0:
                print('affine registration failed %s'%(output_img))
                return output_img

            # run non-rigid
            output_file = os.path.join(temp_dir, source_base_name + '_affine_deeds')
            deeds_deformed_file = self.deeds_reg(target_iso_img, source_iso_img, aff_mtx_file, output_file)

            # run apply deformation to original image
            orgin_res_output_file = os.path.join(temp_dir, source_base_name + '_affine_deeds_origin.nii.gz')
            deeds_deformed_origin_file = self.deeds_apply_origin(target_iso_img, source_img, output_file, aff_mtx_file, orgin_res_output_file)

            cp_cmd = 'cp %s %s' % (deeds_deformed_origin_file, output_img)
            os.system(cp_cmd)
        # rm_cmd = 'rm %s' % (output_img)
        # os.system(rm_cmd)
        return output_img

    def deeds_iso_mask_pipeline(self, target_img, source_img, source_mask, output_img, output_mask, temp_dir):
        if not (os.path.exists(target_img) and os.path.exists(source_img)):
            return output_img

        if not os.path.exists(output_img):
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # get base name
            source_base_name = os.path.basename(source_img)
            source_base_name = source_base_name.replace('.nii.gz', '')
            target_base_name = os.path.basename(target_img)
            target_base_name = target_base_name.replace('.nii.gz', '')

            # #iso resample
            source_iso_img = os.path.join(temp_dir, source_base_name + '_res.nii.gz')
            target_iso_img = os.path.join(temp_dir, target_base_name + '_res.nii.gz')
            self.c3d_resample(source_img, source_iso_img)
            self.c3d_resample(target_img, target_iso_img)

            # # iso padding
            # target_padding_img = os.path.join(temp_dir, target_base_name + '_pad.nii.gz')
            # deeds.c3d_padding(target_iso_img, target_padding_img)

            # run affine
            aff_mtx = os.path.join(temp_dir, source_base_name + '_affine')
            aff_mtx_file = self.affine_reg(target_iso_img, source_iso_img, aff_mtx)

            # see if affine is nan
            text_file = open(aff_mtx_file, 'r')
            lines = text_file.readlines()
            if lines[0].find('nan') > 0:
                print('affine registration failed %s'%(output_img))
                return output_img

            # run non-rigid
            output_file = os.path.join(temp_dir, source_base_name + '_affine_deeds')
            deeds_deformed_file = self.deeds_reg(target_iso_img, source_iso_img, aff_mtx_file, output_file)

            # run apply deformation to original image
            orgin_res_output_file = os.path.join(temp_dir, source_base_name + '_affine_deeds_origin.nii.gz')
            deeds_deformed_origin_file = self.deeds_apply_origin(target_iso_img, source_img, output_file, aff_mtx_file, orgin_res_output_file)

            # run apply deformation to original mask
            orgin_res_output_seg_file = os.path.join(temp_dir, source_base_name + '_affine_deeds_segmentation.nii.gz')
            deeds_deformed_origin_seg_file = self.deeds_apply_origin_seg(target_iso_img, source_mask, output_file, aff_mtx_file, orgin_res_output_seg_file)

            cp_cmd = 'cp %s %s' % (deeds_deformed_origin_file, output_img)
            os.system(cp_cmd)

            cp_cmd = 'cp %s %s' % (orgin_res_output_seg_file, output_mask)
            os.system(cp_cmd)
        # rm_cmd = 'rm %s' % (output_img)
        # os.system(rm_cmd)
        return output_img, output_mask

    def deeds_filter_iso_pipeline(self,target_img, source_img, output_img, temp_dir):
        if not os.path.exists(output_img):
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # get base name
            source_base_name = os.path.basename(source_img)
            source_base_name = source_base_name.replace('.nii.gz', '')
            target_base_name = os.path.basename(target_img)
            target_base_name = target_base_name.replace('.nii.gz', '')

            # filter
            source_filter_img = os.path.join(temp_dir, source_base_name + '_filter.nii.gz')
            target_filter_img = os.path.join(temp_dir, target_base_name + '_filter.nii.gz')
            self.band_filter(source_img, source_filter_img)
            self.band_filter(target_img, target_filter_img)

            # #iso resample
            source_iso_img = os.path.join(temp_dir, source_base_name + '_res.nii.gz')
            target_iso_img = os.path.join(temp_dir, target_base_name + '_res.nii.gz')
            self.c3d_resample(source_filter_img, source_iso_img)
            self.c3d_resample(target_filter_img, target_iso_img)

            # # iso padding
            # target_padding_img = os.path.join(temp_dir, target_base_name + '_pad.nii.gz')
            # deeds.c3d_padding(target_iso_img, target_padding_img)

            # run affine
            aff_mtx = os.path.join(temp_dir, source_base_name + '_affine')
            aff_mtx_file = self.affine_reg(target_iso_img, source_iso_img, aff_mtx)

            # run non-rigid
            output_file = os.path.join(temp_dir, source_base_name + '_affine_deeds')
            deeds_deformed_file = self.deeds_reg(target_iso_img, source_iso_img, aff_mtx_file, output_file)

            cp_cmd = 'cp %s %s' % (deeds_deformed_file,output_img)
            os.system(cp_cmd)
        return output_img

    def inv_affine_mtx(self, affine_mtx, inv_affine_mtx):
        if not os.path.exists(inv_affine_mtx):
            A = np.loadtxt(affine_mtx)
            B = np.linalg.inv(A)
            np.savetxt(inv_affine_mtx, B)

    def deeds_inv_registraiton(self, moving_img, deformation_img, output_img, aff_mtx_file):
        if not os.path.exists(output_img):
            moving_img = '-M %s' % moving_img
            deformation_para = '-O %s' % deformation_img
            output_para = '-D %s' % output_img
            affine_mtx_para = '-A %s' % aff_mtx_file
            cmd = '%s/applyBCVinv %s %s %s %s' % (self.binary_dir, moving_img, deformation_para, output_para, affine_mtx_para)
            os.system(cmd)




if __name__ == "__main__":
    deeds_binary_dir = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline'
    deeds = deeds_registration(binary_dir=deeds_binary_dir)

    target_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/venous_phase.nii.gz'
    temp_dir = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/temp'

    source_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/non-contrast.nii.gz'
    output_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/temp/non-contrast.nii.gz'
    deeds.deeds_iso_pipeline(target_img, source_img, output_img, temp_dir)

    source_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/delay_phase.nii.gz'
    output_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/temp/delay_phase.nii.gz'
    deeds.deeds_iso_pipeline(target_img, source_img, output_img, temp_dir)

    source_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/arterial_phase.nii.gz'
    output_img = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline/temp/arterial_phase.nii.gz'
    deeds.deeds_iso_pipeline(target_img, source_img, output_img, temp_dir)

    # get base name
    source_base_name = os.path.basename(source_img)
    source_base_name = source_base_name.replace('.nii.gz', '')
    target_base_name = os.path.basename(target_img)
    target_base_name = target_base_name.replace('.nii.gz', '')

    aff_mtx_file = os.path.join(temp_dir, 'delay_phase_affine_matrix.txt')
    moving_img = os.path.join(temp_dir, 'delay_phase_affine_deeds_deformed.nii.gz')
    output_img = os.path.join(temp_dir, 'delay_phase_inv_reg.nii.gz')
    deformation_img = os.path.join(temp_dir, 'delay_phase_affine_deeds')
    aff_inv_mtx_file = os.path.join(temp_dir, 'delay_phase_affine_inv_matrix.txt')
    deeds.inv_affine_mtx(aff_mtx_file,aff_inv_mtx_file)
    deeds.deeds_inv_registraiton(moving_img, deformation_img, output_img, aff_inv_mtx_file)




