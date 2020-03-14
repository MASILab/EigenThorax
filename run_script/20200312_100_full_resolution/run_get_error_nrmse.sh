#!/bin/bash

echo "Run on $(whoami)@${HOSTNAME}"

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_100_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020312_100_dataset_full_intens
IN_FOLDER_1=${OUT_ROOT}/pca_20/pc
IN_FOLDER_2=${OUT_ROOT}/pca/pc_batch_2
FILE_LIST_TXT=${OUT_ROOT}/pc_20_list.txt
OUT_PNG_PATH=${OUT_ROOT}/compare_first_20_pc_batch_2.png

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_get_error.py \
  --config ${CONFIG} \
  --in-folder-1 ${IN_FOLDER_1} \
  --in-folder-2 ${IN_FOLDER_2} \
  --file-list-txt ${FILE_LIST_TXT} \
  --out-png-path ${OUT_PNG_PATH}
set +o xtrace


