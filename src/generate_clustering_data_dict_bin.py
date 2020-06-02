import argparse
import numpy as np
from tools.clinical import ClinicalDataReaderSPORE
from tools.data_io import load_object
from tools.utils import get_logger
import datetime
from tools.data_io import save_object


logger = get_logger('ClusterDataDictBin')


def get_attribute_list():
    return [
        'Age', 'sex', 'race', 'ctscannermake', 'heightinches',
        'weightpounds', 'packyearsreported', 'copd', 'Coronary Artery Calcification',
        'cancer_bengin', 'diag_date'
    ]


def generate_data_dict(data_array, label_obj):
    data_dict = {}
    attribute_list = get_attribute_list()
    for data_item in data_array:
        item_dict = {}
        scan_name = data_item['scan_name']

        scan_name_as_record = scan_name
        if not label_obj.check_if_have_record(scan_name):
            # logger.info(f'Cannot find record for {scan_name}')
            scan_name_as_record = label_obj.check_nearest_record_for_impute(scan_name)
            if scan_name_as_record is None:
                continue

        for attr in attribute_list:
            item_dict[attr] = label_obj.get_value_field(scan_name_as_record, attr)

        item_dict['CAC'] = item_dict['Coronary Artery Calcification']
        item_dict['Cancer'] = item_dict['cancer_bengin']
        item_dict['COPD'] = item_dict['copd']
        item_dict['Packyear'] = item_dict['packyearsreported']
        item_dict['SubjectID'] = label_obj._get_subject_id_from_file_name(scan_name)
        item_dict['ScanDate'] = label_obj._get_date_str_from_file_name(scan_name)
        if item_dict['Cancer'] == 1:
            scan_date_obj = ClinicalDataReaderSPORE._get_date_str_from_file_name(scan_name)
            diag_date_obj = datetime.datetime.strptime(str(int(item_dict['diag_date'])), '%Y%m%d')
            time_2_diag = diag_date_obj - scan_date_obj
            item_dict['Time2Diag'] = time_2_diag
            if time_2_diag >= datetime.timedelta(days=365):
                logger.info(time_2_diag)
            item_dict['CancerIncubation'] = int(time_2_diag >= datetime.timedelta(days=365))
            item_dict['CancerSubjectFirstScan'] = label_obj.is_first_cancer_scan(scan_name)

        # BMI = mass(lb)/height(inch)^2 * 703
        bmi_val = np.nan
        mass_lb = item_dict['weightpounds']
        height_inch = item_dict['heightinches']
        if (70 < mass_lb < 400) and (40 < height_inch < 90):
            bmi_val = 703 * mass_lb / (height_inch * height_inch)
        item_dict['bmi'] = bmi_val

        # Image data
        item_dict['ImageData'] = data_item['low_dim']

        data_dict[scan_name] = item_dict

    return data_dict


def main():
    parser = argparse.ArgumentParser(description='Load a saved pca object')
    parser.add_argument('--in-pca-data-bin', type=str)
    parser.add_argument('--label-file', type=str)
    parser.add_argument('--out-data-dict-bin', type=str)
    args = parser.parse_args()

    low_dim_array = load_object(args.in_pca_data_bin)
    label_obj = ClinicalDataReaderSPORE.create_spore_data_reader_xlsx(args.label_file)
    data_dict = generate_data_dict(low_dim_array, label_obj)
    logger.info(f'Save dict data object to {args.out_data_dict_bin}')
    save_object(data_dict, args.out_data_dict_bin)


if __name__ == '__main__':
    main()