#!/bin/bash

SRC_ROOT=/src/thorax_pca
PYTHON_ENV=/opt/conda/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_vlsp_singularity.yaml
OUT_ROOT=/out
UNION_AVERAGE_IMG=${OUT_ROOT}/union_ave.nii.gz
OUT_IMPUTE_FOLDER=${OUT_ROOT}/preprocess/imputation


echo "#####################"
echo "# Run PCA"
PCA_ROOT=${OUT_ROOT}/pca_n_patch_20
OUT_PC_FOLDER=${PCA_ROOT}/pc
OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz
N_COMPONENTS=20
N_BATCH=20
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

