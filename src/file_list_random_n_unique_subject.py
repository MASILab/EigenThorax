import argparse
from tools.utils import get_logger
from tools.utils import read_file_contents_list, save_file_contents_list
import random
from tools.clinical import ClinicalDataReaderSPORE


logger = get_logger('Random select')


def main():
    parser = argparse.ArgumentParser('Plot box and scatter data.')
    parser.add_argument('--file-list-total', type=str)
    parser.add_argument('--file-list-out', type=str)
    parser.add_argument('--num-file-select', type=int)

    args = parser.parse_args()

    file_list_total = read_file_contents_list(args.file_list_total)

    subject_list_total, subject_list_unique = get_subject_id_list(file_list_total)
    logger.info(f'num of total files {len(subject_list_total)}')
    logger.info(f'num of unique files {len(unique(subject_list_total))}')
    logger.info(f'num of unique subject {len(subject_list_unique)}')

    # selected_subject_list = random.choices(subject_list_unique, k=args.num_file_select)
    selected_subject_list = random.sample(subject_list_unique, args.num_file_select)
    logger.info(f'num of selected subjects {len(selected_subject_list)}')
    logger.info(f'num of unique selected subjects {len(unique(selected_subject_list))}')

    file_list_out = [
        file_list_total[subject_list_total.index(subject_id)]
        for subject_id in selected_subject_list
    ]

    save_file_contents_list(args.file_list_out, file_list_out)

def get_subject_id_list(subject_id_exclude_list):
    total_list = [
        ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
        for file_name in subject_id_exclude_list]

    return total_list, unique(total_list)

def unique(in_list):
    # insert the list to the set
    list_set = set(in_list)
    # convert the set to the list
    unique_list = (list(list_set))

    return unique_list


if __name__ == '__main__':
    main()
