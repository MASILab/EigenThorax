#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_100_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020312_100_dataset
IN_FOLDER=${OUT_ROOT}/preprocess/imputation
PCA_ROOT=${OUT_ROOT}/pca
OUT_PC_FOLDER=${PCA_ROOT}/pc
OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz
REF_IMG=${OUT_ROOT}/average/union.nii.gz
N_COMPONENTS=6

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


