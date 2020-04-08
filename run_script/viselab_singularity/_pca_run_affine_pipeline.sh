#!/bin/bash

SRC_ROOT=/src/thorax_pca
PYTHON_ENV=/opt/conda/envs/python37/bin/python

CONFIG=/proj_root/config.sh
source ${CONFIG}

CONFIG_YAML=${SRC_ROOT}/config/pca_vlsp_singularity.yaml
PROJ_ROOT=/proj_root
IN_DATA_FOLDER=/data_root
FILE_LIST=/proj_root/data/file_list
REF_IMG=/proj_root/reference/roi_mask.nii.gz

OUTPUT_ROOT=${PROJ_ROOT}/output
PREPROCESS_ROOT=${OUTPUT_ROOT}/preprocess
UNION_AVERAGE_IMG=${PREPROCESS_ROOT}/union_average.nii.gz

mkdir -p ${OUTPUT_ROOT}
mkdir -p ${PREPROCESS_ROOT}

echo "#####################"
echo "# Step.1 Generate union average. We don't need to resample. Use the average atlas as reference."
set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/generate_union_average.py \
  --config ${CONFIG_YAML} \
  --in-folder ${IN_DATA_FOLDER} \
  --save-path ${UNION_AVERAGE_IMG} \
  --data-file-list ${FILE_LIST}
set +o xtrace
echo

echo "#####################"
echo "# Step.2 Do average imputation on the original scans using the union average."
OUT_IMPUTE_FOLDER=${PREPROCESS_ROOT}/imputation
mkdir -p ${OUT_IMPUTE_FOLDER}

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_average_imputation.py \
  --config ${CONFIG_YAML} \
  --in-folder ${IN_DATA_FOLDER} \
  --out-folder ${OUT_IMPUTE_FOLDER} \
  --average-img ${UNION_AVERAGE_IMG} \
  --data-file-list ${FILE_LIST}
set +o xtrace
echo

echo "#####################"
echo "# Step.3 Run PCA"
PCA_ROOT=${OUTPUT_ROOT}/pca
OUT_PC_FOLDER=${PCA_ROOT}/pc
OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz

OUT_PCA_BIN_PATH=${PCA_ROOT}/model.bin
mkdir -p ${OUT_PC_FOLDER}

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_pca_batch.py \
  --config ${CONFIG_YAML} \
  --in-folder ${OUT_IMPUTE_FOLDER} \
  --in-file-list ${FILE_LIST} \
  --ref-img ${REF_IMG} \
  --out-pc-folder ${OUT_PC_FOLDER} \
  --out-mean-img ${OUT_MEAN_IMG} \
  --n-components "${N_COMPONENTS}" \
  --batch-size "${BATCH_SIZE}" \
  --save-pca-result-path ${OUT_PCA_BIN_PATH}
set +o xtrace

echo "#####################"
echo "# Step.4 Plot PCs and scree"
ANALYSIS_ROOT=${PROJ_ROOT}/analysis
mkdir -p ${ANALYSIS_ROOT}
PC_FILE_LIST=${PCA_ROOT}/pc_file_list.txt
ls ${OUT_PC_FOLDER} >> ${PC_FILE_LIST}

N_PC_PLOT_PNG=${ANALYSIS_ROOT}/10pc.png
SCREE_PLOT_PNG=${ANALYSIS_ROOT}/scree.png

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/plot_first_n_pc.py \
  --pc-folder ${OUT_PC_FOLDER} \
  --file-list-txt ${PC_FILE_LIST} \
  --out-png ${N_PC_PLOT_PNG}
set +o xtrace

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/plot_pca_scree.py \
  --load-pca-bin-path ${OUT_PCA_BIN_PATH} \
  --save-img-path ${SCREE_PLOT_PNG}
set +o xtrace





