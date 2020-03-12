#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_100_data.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020312_100_dataset
IN_FOLDER=/nfs/masi/xuk9/SPORE/registration/affine_vlsp/20200308_vlsp_config_v4/20200308_vlsp_config_v4/interp/ori_100
OUT_FOLDER=${OUT_ROOT}/preprocess/resample
REF_IMG=/nfs/masi/xuk9/SPORE/clustering/pca/2020311_10_dataset/average/union.nii.gz

mkdir -p ${OUT_FOLDER}
set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_resample.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --out-folder ${OUT_FOLDER} \
  --ref-img ${REF_IMG}
set +o xtrace


