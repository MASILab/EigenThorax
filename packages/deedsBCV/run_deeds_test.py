import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from deeds_registration import deeds_registration
import glob

if __name__ == '__main__':

    # set up path

    source_datafolder_path = '/media/yuankai/Data/project/CT_liver/Yuankai_process/test/inputs/'
    save_datafolder_path = '/media/yuankai/Data/project/CT_liver/Yuankai_process/test/outputs'
    # read xlsx
    flist = os.listdir(source_datafolder_path)

    SID = np.unique(flist)

    n = 0

    deeds_binary_dir = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline'
    deeds = deeds_registration(binary_dir=deeds_binary_dir)

    for study in range(len(SID)):


        study_id = SID[study]


        input_NC = os.path.join(source_datafolder_path, study_id, 'non-contrast.nii.gz')
        input_A = os.path.join(source_datafolder_path, study_id, 'arterial_phase.nii.gz')
        input_V = os.path.join(source_datafolder_path, study_id, 'venous_phase.nii.gz')
        input_D = os.path.join(source_datafolder_path, study_id, 'delay_phase.nii.gz')

        # save in NII format with filename=phase name
        save_path = os.path.join(save_datafolder_path, study_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        temp_dir = os.path.join(save_path, 'temp')

        save_filename_NCcrop = 'non-contrast.nii.gz'
        save_filename_Acrop = 'arterial_phase.nii.gz'
        save_filename_Vcrop = 'venous_phase.nii.gz'
        save_filename_Dcrop = 'delay_phase.nii.gz'
        output_NC = os.path.join(save_path, save_filename_NCcrop)
        output_A = os.path.join(save_path, save_filename_Acrop)
        output_V = os.path.join(save_path, save_filename_Vcrop)
        output_D = os.path.join(save_path, save_filename_Dcrop)

        try:
            deeds.deeds_iso_pipeline(input_A, input_V, output_V, temp_dir)
        except:
            print('Fail:' + SID[study])


        try:
            if not os.path.exists(output_A):
                cp_cmd = 'cp %s %s' % (input_A, output_A)
                os.system(cp_cmd)
        except:
            print('Fail:' + SID[study])

        print('done %d/%d'%(study,len(SID)))

