#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_missing_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020311_10_dataset
IN_FOLDER=${OUT_ROOT}/imputation/average_imputation
OUT_PC_FOLDER=${OUT_ROOT}/PCA/PCs
OUT_MEAN_IMG=${OUT_ROOT}/PCA/MEAN/mean.nii.gz
REF_IMG=${OUT_ROOT}/average/union.nii.gz
N_COMPONENTS=3

mkdir -p ${OUT_PC_FOLDER}

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_pca.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --ref-img ${REF_IMG} \
  --out-pc-folder ${OUT_PC_FOLDER} \
  --out-mean-img ${OUT_MEAN_IMG} \
  --n-components "${N_COMPONENTS}"
set +o xtrace


