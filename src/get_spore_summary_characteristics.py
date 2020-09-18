import numpy as np
from os import path
from tools.utils import get_logger
import pandas as pd
from tools.data_io import load_object, read_file_contents_list
from tools.clinical_spore_summary_characteristics import ClinicalDataReaderSPORE


spore_csv = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/clinical/label_full.csv'
in_feature_matrix_bin = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/pca_amazon_server/result/pca_ori_400/dr_data.bin'
ori_spore_excel = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/clinical/label.xlsx'

female_file_list ='/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/female/data/success_list_include_cancer'

def main():
    file_list = load_object(in_feature_matrix_bin)['file_list']
    # file_list = read_file_contents_list(female_file_list)
    subject_list = ClinicalDataReaderSPORE.get_subject_list(file_list)

    reader_obj = ClinicalDataReaderSPORE.create_spore_data_reader_csv(spore_csv)
    ori_spore_label_df = pd.read_excel(ori_spore_excel)
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'copd')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'Coronary Artery Calcification')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'race')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'LungRADS')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'smokingstatus')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'packyearsreported')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'education')
    reader_obj.get_attributes_from_original_label_file(ori_spore_label_df, 'cancer_bengin')

    reader_obj.get_summary_characteristics_subject(subject_list)


if __name__ == '__main__':
    main()