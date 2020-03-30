#!/bin/bash

SRC_ROOT=/src/thorax_pca
PYTHON_ENV=/opt/conda/envs/python37/bin/python

CONFIG=/proj_root/config.sh
source ${CONFIG}

CONFIG_YAML=${SRC_ROOT}/config/pca_vlsp_singularity.yaml
OUT_ROOT=/proj_root
IN_DATA_FOLDER=/data_root
FILE_LIST=/proj_root/data/file_list
REF_IMG=/proj_root/reference/roi_mask.nii.gz

echo "#####################"
echo "# Run PCA"
PCA_ROOT=${OUT_ROOT}/pca
OUT_PC_FOLDER=${PCA_ROOT}/pc
OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz

OUT_PCA_BIN_PATH=${PCA_ROOT}/model.bin
mkdir -p ${OUT_PC_FOLDER}

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_pca_batch.py \
  --config ${CONFIG_YAML} \
  --in-folder ${IN_DATA_FOLDER} \
  --in-file-list ${FILE_LIST} \
  --ref-img ${REF_IMG} \
  --out-pc-folder ${OUT_PC_FOLDER} \
  --out-mean-img ${OUT_MEAN_IMG} \
  --n-components "${N_COMPONENTS}" \
  --batch-size "${BATCH_SIZE}" \
  --save-pca-result-path ${OUT_PCA_BIN_PATH}
set +o xtrace

