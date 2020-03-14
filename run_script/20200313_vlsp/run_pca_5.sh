#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_vlsp.yaml
OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020313_vlsp
IN_FOLDER=/nfs/masi/xuk9/SPORE/registration/affine_vlsp/20200308_vlsp_config_v4/20200308_vlsp_config_v4/interp/ori
UNION_AVERAGE_IMG=${OUT_ROOT}/union_ave.nii.gz

echo "#####################"
echo "# Step.1 Generate union average. We don't need to resample. Use the average atlas as reference."
mkdir -p ${OUT_ROOT}
#set -o xtrace
#${PYTHON_ENV} ${SRC_ROOT}/src/generate_union_average.py \
#  --config ${CONFIG} \
#  --in-folder ${IN_FOLDER} \
#  --save-path ${UNION_AVERAGE_IMG}
#set +o xtrace
echo

echo "#####################"
echo "# Step.2 Do average imputation on the original scans using the union average."
OUT_IMPUTE_FOLDER=${OUT_ROOT}/preprocess/imputation
mkdir -p ${OUT_IMPUTE_FOLDER}

#set -o xtrace
#${PYTHON_ENV} ${SRC_ROOT}/src/run_average_imputation.py \
#  --config ${CONFIG} \
#  --in-folder ${IN_FOLDER} \
#  --out-folder ${OUT_IMPUTE_FOLDER} \
#  --average-img ${UNION_AVERAGE_IMG}
#set +o xtrace
echo

echo "#####################"
echo "# Step.3 Run PCA"
PCA_ROOT=${OUT_ROOT}/pca_4
OUT_PC_FOLDER=${PCA_ROOT}/pc_20
OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz
N_COMPONENTS=20
N_BATCH=40
OUT_PCA_BIN_PATH=${PCA_ROOT}/model.bin
mkdir -p ${OUT_PC_FOLDER}

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_pca_batch.py \
  --config ${CONFIG} \
  --in-folder ${OUT_IMPUTE_FOLDER} \
  --ref-img ${UNION_AVERAGE_IMG} \
  --out-pc-folder ${OUT_PC_FOLDER} \
  --out-mean-img ${OUT_MEAN_IMG} \
  --n-components "${N_COMPONENTS}" \
  --n-batch "$N_BATCH" \
  --save-pca-result-path ${OUT_PCA_BIN_PATH}
set +o xtrace

