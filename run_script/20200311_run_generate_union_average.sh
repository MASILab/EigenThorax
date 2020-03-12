#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_missing_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020311_10_dataset
IN_FOLDER=${OUT_ROOT}/data_preprocess_niftyreg
SAVE_PATH=${OUT_ROOT}/average/union.nii.gz

REF_IMG=/nfs/masi/xuk9/SPORE/clustering/pca/2020311_10_dataset/data_preprocess/00000103time20170511.nii.gz

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/generate_union_average.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --save-path ${SAVE_PATH}
#${PYTHON_ENV} ${SRC_ROOT}/src/tools/average_images_with_nan.py \
#  --in_folder ${IN_FOLDER} \
#  --out_union ${SAVE_PATH} \
#  --ref ${REF_IMG} \
#  --num_processes 3
set +o xtrace


