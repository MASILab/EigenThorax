import argparse
from tools.classifier import MinibatchLinearClassifierWithCV
from tools.utils import read_file_contents_list, mkdir_p
from tools.clinical_spore_summary_characteristics import ClinicalDataReaderSPORE
import pandas as pd
from tools.data_io import save_object, load_object
from os import path


# in_folder = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/pca_concat/data/jac/trimed'
in_folder = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/pca_concat/data/jac/trimed_downsampled'
file_list_txt = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/data/success_list_include_cancer'
in_csv_file = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/clinical/label_full.csv'
proj_folder = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/hypothesis_2/train_classifier'

mkdir_p(proj_folder)

num_fold = 5
batch_size = 10

if_run_training = False
if_run_validation = True

def main():
    file_list = read_file_contents_list(file_list_txt)

    clinical_data_reader = ClinicalDataReaderSPORE.create_spore_data_reader_csv(in_csv_file)
    label_list = clinical_data_reader.get_label_for_obese(file_list)
    data_tuples = list(zip(file_list, label_list))
    label_df = pd.DataFrame(data_tuples, columns=['scan', 'label'])

    classifier_obj = MinibatchLinearClassifierWithCV.create_classifier_obj(
        in_folder,
        file_list,
        num_fold,
        label_df,
        batch_size
    )

    save_bin_path = path.join(proj_folder, 'model.bin')
    if if_run_training:
        classifier_obj.train()
        classifier_obj.validate()
        # classifier_obj.train_first_fold()
        # save_object(classifier_obj, save_bin_path)

    if if_run_validation:
        classifier_obj = load_object(save_bin_path)
        classifier_obj.valid_first_fold()
        auc_roc_first_fold = classifier_obj.validation_result[0]['roc_auc']
        print(f'auc_roc of fold 0: {auc_roc_first_fold}')

    # for idx_fold in range(num_fold):
    #     auc_roc = classifier_obj.validation_result[idx_fold]['roc_auc']
    #     print(f'AOC_ROC of fold {idx_fold}: {auc_roc}')


if __name__ == '__main__':
    main()