import argparse
from tools.classifier import MinibatchLinearClassifierWithCV
from tools.utils import read_file_contents_list


in_folder = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/pca_concat/data/jac/trimed'
file_list_txt = '/nfs/masi/xuk9/SPORE/clustering/registration/20200512_corrField/male/data/success_list_include_cancer'

num_fold = 5
batch_size = 10


def main():
    file_list = read_file_contents_list(file_list_txt)

    classifier_obj = MinibatchLinearClassifierWithCV.create_classifier_obj(
        in_folder,
        file_list,
        num_fold,

    )


if __name__ == '__main__':
    main()