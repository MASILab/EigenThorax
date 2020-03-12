#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_100_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020312_100_dataset
IN_FOLDER=${OUT_ROOT}/preprocess/resample
SAVE_PATH=${OUT_ROOT}/average/union.nii.gz

REF_IMG=/nfs/masi/xuk9/SPORE/clustering/pca/2020311_10_dataset/data_preprocess/00000103time20170511.nii.gz

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/generate_union_average.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --save-path ${SAVE_PATH}
set +o xtrace
