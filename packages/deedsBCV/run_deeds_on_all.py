import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from deeds_registration import deeds_registration

if __name__ == '__main__':

    # set up path
    source_datafolder_path = '/media/yuankai/Data/project/CT_liver/raw_from_bo/allmining/liver-ct-allmining/data_preprocessing/data/data_step2/'
    save_datafolder_path = '/media/yuankai/Data/project/CT_liver/Yuankai_process/liver-ct-allmining/data_preprocessing/data_step3/'
    source_infofile_path1 = '/media/yuankai/Data/project/CT_liver/raw_from_bo/allmining/liver-ct-allmining/data_preprocessing/lists/processed_step1/train_step1.xlsx'
    source_infofile_path2 = '/media/yuankai/Data/project/CT_liver/raw_from_bo/allmining/liver-ct-allmining/data_preprocessing/lists/processed_step1/vali_step1.xlsx'
    source_infofile_path3 = '/media/yuankai/Data/project/CT_liver/raw_from_bo/allmining/liver-ct-allmining/data_preprocessing/lists/processed_step1/test_step1.xlsx'
    # read xlsx
    df1 = pd.read_excel(source_infofile_path1)
    df1 = df1.drop('Unnamed: 0', axis=1)
    SID1 = np.unique(df1["study_id"])

    df2 = pd.read_excel(source_infofile_path2)
    df2 = df2.drop('Unnamed: 0', axis=1)
    SID2 = np.unique(df2["study_id"])

    df3 = pd.read_excel(source_infofile_path3)
    df3 = df3.drop('Unnamed: 0', axis=1)
    SID3 = np.unique(df3["study_id"])

    SID = np.concatenate((SID1,SID2,SID3), axis=0)

    n = 0

    deeds_binary_dir = '/home/yuankai/Projects/InnerCircle/Deeds/python_pipeline'
    deeds = deeds_registration(binary_dir=deeds_binary_dir)

    for study in range(len(SID)):

        study_dat1 = df1.loc[df1["study_id"] == SID[study]]
        study_dat2 = df2.loc[df2["study_id"] == SID[study]]
        study_dat3 = df3.loc[df3["study_id"] == SID[study]]

        study_dat = pd.concat([study_dat1, study_dat2, study_dat3])

        study_id = np.array(study_dat.iloc[:, 0])[0]
        patient_id = np.array(study_dat.iloc[:, 1])[0]
        patho_id = np.array(study_dat.iloc[:, 2])[0]


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
            deeds.deeds_iso_pipeline(input_V, input_NC, output_NC, temp_dir)
        except:
            print('Fail:' + SID[study])
        try:
            deeds.deeds_iso_pipeline(input_V, input_A, output_A, temp_dir)
        except:
            print('Fail:' + SID[study])
        try:
            deeds.deeds_iso_pipeline(input_V, input_D, output_D, temp_dir)
        except:
            print('Fail:' + SID[study])

        try:
            if not os.path.exists(output_V):
                temp_V =  os.path.join(temp_dir, 'venous_phase.nii.gz')
                if os.path.exists(input_V):
                    cp_cmd = 'cp %s %s' % (input_V, output_V)
                    os.system(cp_cmd)
            # else:
            #     rm_cmd = 'rm %s' % (output_V)
            #     os.system(rm_cmd)
        except:
            print('Fail:' + SID[study])

        print('done %d/%d'%(study,len(SID)))

