#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/20200315_vlsp_gender
CLINICAL_LABEL_XLSX=${OUT_ROOT}/combine_17and18.xlsx
TOTAL_FILE_LIST=${OUT_ROOT}/vlsp_1473.txt

OUT_FILE_LIST_ROOT=${OUT_ROOT}/file_list
OUT_MALE_TXT=${OUT_FILE_LIST_ROOT}/male.txt
OUT_FEMALE_TXT=${OUT_FILE_LIST_ROOT}/female.txt

PC_FILE_LIST=${OUT_ROOT}/pc_file_list_10.txt

MALE_FLAG="male"
FEMALE_FLAG="female"

get_pc_fig_gender () {
  local GENDER_FLAG=$1

  local GENDER_ROOT=${OUT_ROOT}/${GENDER_FLAG}
  mkdir -p ${GENDER_ROOT}

  PCA_ROOT=${GENDER_ROOT}/pca
  OUT_PC_FOLDER=${PCA_ROOT}/pc
  OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz
  OUT_PNG_FOLDER=${PCA_ROOT}/pc_fig
#  OUT_PNG=${PCA_ROOT}/10pc.png

  set -o xtrace
  ${PYTHON_ENV} ${SRC_ROOT}/src/plot_first_n_pc.py \
    --pc-folder ${OUT_PC_FOLDER} \
    --file-list-txt ${PC_FILE_LIST} \
    --out-png-folder ${OUT_PNG_FOLDER}
#    --out-png ${OUT_PNG}
  set +o xtrace
}

get_pc_fig_gender ${MALE_FLAG}
get_pc_fig_gender ${FEMALE_FLAG}