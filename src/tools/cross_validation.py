from tools.utils import get_logger
from tools.clinical import ClinicalDataReaderSPORE
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np


logger = get_logger('Cross-validation utilities.')


def get_idx_list_array_n_fold_cross_validation(file_name_list, label_list, num_fold):
    """
    Get the n-folder split at subject level (scans of the same subject always go into one fold)
    :param file_name_list: file name list of scans, with .nii.gz
    :param num_fold: number of folds
    :return:
    """
    scan_label = label_list
    subject_id_full = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
                       for file_name in file_name_list]
    subject_id_unique = list(set(subject_id_full))
    subject_label = [label_list[subject_id_full.index(subject_id)]
                     for subject_id in subject_id_unique]

    skf = StratifiedKFold(n_splits=num_fold, random_state=0)
    # skf = KFold(n_splits=num_fold, random_state=0)
    logger.info(f'Split data set into {skf.get_n_splits()} folds.')
    logger.info(f'Number of scans: {len(file_name_list)}')
    logger.info(f'Number of subjects: {len(subject_id_unique)}')

    subject_train_idx_list_array = []
    subject_test_idx_list_array = []
    for train_idx_list, test_idx_list in skf.split(subject_id_unique, subject_label):
        subject_train_idx_list_array.append(train_idx_list)
        subject_test_idx_list_array.append(test_idx_list)

    # for train_idx_list, test_idx_list in skf.split(subject_id_unique):
    #     subject_train_idx_list_array.append(train_idx_list)
    #     subject_test_idx_list_array.append(test_idx_list)

    scan_train_idx_list_array = []
    scan_test_idx_list_array = []
    for idx_fold in range(num_fold):
        scan_train_idx_list = []
        scan_test_idx_list = []
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        subject_test_idx_list = subject_test_idx_list_array[idx_fold]

        for idx_subject in subject_train_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_train_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_train_idx_list += subject_scan_train_idx_list

        for idx_subject in subject_test_idx_list:
            subject_id = subject_id_unique[idx_subject]
            subject_scan_test_idx_list = [idx for idx, subject in enumerate(subject_id_full) if subject == subject_id]
            scan_test_idx_list += subject_scan_test_idx_list

        scan_train_idx_list_array.append(scan_train_idx_list)
        scan_test_idx_list_array.append(scan_test_idx_list)

    num_pos_scan_train_fold_array = []
    num_pos_scan_test_fold_array = []
    num_pos_subject_train_fold_array = []
    num_pos_subject_test_fold_array = []

    for idx_fold in range(num_fold):
        scan_train_idx_list = scan_train_idx_list_array[idx_fold]
        scan_test_idx_list = scan_test_idx_list_array[idx_fold]
        subject_train_idx_list = subject_train_idx_list_array[idx_fold]
        subject_test_idx_list = subject_test_idx_list_array[idx_fold]

        num_pos_scan_train_fold_array.append(
            np.sum(np.array([scan_label[idx] for idx in scan_train_idx_list])))
        num_pos_scan_test_fold_array.append(
            np.sum(np.array([scan_label[idx] for idx in scan_test_idx_list])))
        # print([file_name_list[idx] for idx in scan_test_idx_list if scan_label[idx] == 1])
        num_pos_subject_train_fold_array.append(
            np.sum(np.array([subject_label[idx] for idx in subject_train_idx_list])))
        num_pos_subject_test_fold_array.append(
            np.sum(np.array([subject_label[idx] for idx in subject_test_idx_list])))

    logger.info(f'Sizes of each fold:')
    logger.info(f'Train (subject): {[len(train_subject_list) for train_subject_list in subject_train_idx_list_array]}')
    logger.info(f'Test (subject): {[len(test_subject_list) for test_subject_list in subject_test_idx_list_array]}')
    logger.info(f'Train (scan): {[len(train_list) for train_list in scan_train_idx_list_array]}')
    logger.info(f'Test (scan): {[len(test_list) for test_list in scan_test_idx_list_array]}')
    logger.info(f'Train-pos (subject): {num_pos_subject_train_fold_array}')
    logger.info(f'Train-pos (scan): {num_pos_scan_train_fold_array}')
    logger.info(f'Test-pos (subject): {num_pos_subject_test_fold_array}')
    logger.info(f'Test-pos (scan): {num_pos_scan_test_fold_array}')

    return scan_train_idx_list_array, scan_test_idx_list_array

