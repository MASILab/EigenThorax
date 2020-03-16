#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/20200315_vlsp_gender
CLINICAL_LABEL_XLSX=${OUT_ROOT}/combine_17and18.xlsx
TOTAL_FILE_LIST=${OUT_ROOT}/vlsp_1473.txt

OUT_FILE_LIST_ROOT=${OUT_ROOT}/file_list
OUT_MALE_TXT=${OUT_FILE_LIST_ROOT}/male.txt
OUT_FEMALE_TXT=${OUT_FILE_LIST_ROOT}/female.txt

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/get_gender_file_list.py \
  --total-file-list ${TOTAL_FILE_LIST} \
  --clinical-label-xlsx ${CLINICAL_LABEL_XLSX} \
  --gender-str "M" \
  --out-file-list-txt ${OUT_MALE_TXT}

${PYTHON_ENV} ${SRC_ROOT}/src/get_gender_file_list.py \
  --total-file-list ${TOTAL_FILE_LIST} \
  --clinical-label-xlsx ${CLINICAL_LABEL_XLSX} \
  --gender-str "F" \
  --out-file-list-txt ${OUT_FEMALE_TXT}
set +o xtrace
