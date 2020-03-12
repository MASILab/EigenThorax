#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_100_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020312_100_dataset
IN_FOLDER=${OUT_ROOT}/preprocess/resample
OUT_FOLDER=${OUT_ROOT}/preprocess/imputation
AVE_IMG=${OUT_ROOT}/average/union.nii.gz

mkdir -p ${OUT_FOLDER}
set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_average_imputation.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --out-folder ${OUT_FOLDER} \
  --average-img ${AVE_IMG}
set +o xtrace

