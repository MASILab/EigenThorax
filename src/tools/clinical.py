import pandas as pd
import re


class ClinicalDataReaderSPORE:
    def __init__(self, data_frame):
        self._df = data_frame

    def filter_sublist_with_label(self, in_file_list, field_name, field_flag):
        print(f'Filtering file list with field_name ({field_name}) and flag ({field_flag})', flush=True)
        out_file_list = []
        for file_name in in_file_list:
            subject_id = self._get_subject_id_from_file_name(file_name)
            if self._if_field_match(subject_id, field_name, field_flag):
                out_file_list.append(file_name)

        print(f'Complete. Find {len(out_file_list)} matching items.')
        return out_file_list

    def _if_field_match(self, subj, field_name, field_flag):
        subj_name = self._get_name_field_flat_from_sub_id(subj)
        row_id_list = self._df.loc[self._df['sub_name'] == subj_name]

        if_match = False
        if len(row_id_list) == 0:
            print(f'Cannot find subject id {subj_name}', flush=True)
        else:
            row_id = self._df.loc[self._df['sub_name'] == subj_name].index[0]
            sex_str = str(self._df.at[row_id, field_name])
            # sex_str = str(self._df.get_value(row_id, field_name))
            if_match = sex_str == field_flag

        return if_match

    @staticmethod
    def _get_subject_id_from_file_name(file_name_nii_gz):
        match_list = re.match(r"(?P<subject_id>\d+)time(?P<time_id>\d+).nii.gz", file_name_nii_gz)
        subject_id = int(match_list.group('subject_id'))
        return subject_id

    @staticmethod
    def _get_name_field_flat_from_sub_id(sub_id):
        return f'SPORE_{sub_id:08}'

    @staticmethod
    def create_spore_data_reader_xlsx(file_xlsx):
        return ClinicalDataReaderSPORE(pd.read_excel(file_xlsx))
