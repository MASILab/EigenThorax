import pandas as pd
import re
from datetime import datetime
from tools.utils import get_logger
import numpy as np

logger = get_logger('Clinical')


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

    def check_if_have_record(self, file_name_nii_gz):
        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)
        filtered_rows = \
            self._df[(self._df['SPORE'] == spore_name_field) & (self._df['studydate'] == np.datetime64(spore_date_field))]

        count_row = filtered_rows.shape[0]

        return count_row != 0

    def check_nearest_record_for_impute(self, file_name_nii_gz):
        cp_record_file_name = None

        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)
        spore_date_field = np.datetime64(spore_date_field)
        filtered_rows_name_field = \
            self._df[(self._df['SPORE'] == spore_name_field)]

        if filtered_rows_name_field.shape[0] > 0:
            time_list = filtered_rows_name_field['studydate'].to_numpy()
            time_delta_list = time_list - spore_date_field
            time_days_list = time_delta_list / np.timedelta64(1, 'D')
            time_days_list = np.abs(time_days_list)
            min_idx = np.argmin(time_days_list)

            subj_id = self._get_subject_id_from_file_name(file_name_nii_gz)
            time_closest = time_list[min_idx]
            datetime_obj = pd.to_datetime(str(time_closest))
            datetime_str = datetime_obj.strftime('%Y%m%d')
            cp_record_file_name = f'{subj_id:08}time{datetime_str}.nii.gz'

        return cp_record_file_name

    def get_value_field(self, file_name_nii_gz, field_flag):
        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)

        filtered_rows = \
            self._df[(self._df['SPORE'] == spore_name_field) & (self._df['studydate'] == np.datetime64(spore_date_field))]

        return_val = np.nan
        count_row = filtered_rows.shape[0]
        if count_row == 0:
            logger.info(f'Cannot find label item for {file_name_nii_gz}')
        else:
            return_val = filtered_rows.iloc[0][field_flag]
            # logger.info(f'Field value: {return_val}')

        return return_val

    @staticmethod
    def _get_date_str_from_file_name(file_name_nii_gz):
        match_list = re.match(r"(?P<subject_id>\d+)time(?P<time_id>\d+).nii.gz", file_name_nii_gz)
        date_str_ori = match_list.group('time_id')
        date_obj = datetime.strptime(date_str_ori, "%Y%m%d")
        # date_str = date_obj.strftime("%m/%d/%Y")
        return date_obj

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
