import argparse
from tools.utils import get_logger
from tools.utils import read_file_contents_list, save_file_contents_list
from tools.clinical import ClinicalDataReaderSPORE

logger = get_logger('Exclude file list')


def main():
    parser = argparse.ArgumentParser('Plot box and scatter data.')
    parser.add_argument('--file-list-total', type=str)
    parser.add_argument('--subject-id-exclude-file-list', type=str)
    parser.add_argument('--file-list-out', type=str)
    args = parser.parse_args()

    file_list_total = read_file_contents_list(args.file_list_total)
    subject_id_exclude_file_list = read_file_contents_list(args.subject_id_exclude_file_list)

    subject_id_exclude_list = get_subject_id_list(subject_id_exclude_file_list)

    file_list_reduced = [
        file_name for
        file_name in file_list_total
        if ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
           not in subject_id_exclude_list]

    save_file_contents_list(args.file_list_out, file_list_reduced)


def get_subject_id_list(subject_id_exclude_list):
    subject_id_list_to_exclude = [
        ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
        for file_name in subject_id_exclude_list]

    return unique(subject_id_list_to_exclude)

def unique(in_list):
    # insert the list to the set
    list_set = set(in_list)
    # convert the set to the list
    unique_list = (list(list_set))

    return unique_list


if __name__ == '__main__':
    main()
