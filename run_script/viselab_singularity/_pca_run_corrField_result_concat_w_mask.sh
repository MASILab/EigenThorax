#!/bin/bash

########
# config.sh example
########
#SRC_ROOT=/src/thorax_pca
#PYTHON_ENV=/opt/conda/envs/python37/bin/python
#
#PROJ_ROOT=/proj_root
#IN_DATA_FOLDER=/data_root/analysis/data/non_rigid/imputed_folder
#CONFIG_YAML=${SRC_ROOT}/config/pca_vlsp_singularity.yaml
#
#FILE_LIST=/data_root/data/success_list
#REF_IMG=/data_root/analysis/data/non_rigid/average/average.nii.gz
#
#N_COMPONENTS=20
#BATCH_SIZE=80

CONFIG=/proj_root/config.sh
source ${CONFIG}

OUTPUT_ROOT=${PROJ_ROOT}/output

mkdir -p ${OUTPUT_ROOT}

echo "#####################"
echo "# Run PCA"
PCA_ROOT=${OUTPUT_ROOT}
OUT_PC_FOLDER=${PCA_ROOT}/pc
#OUT_MEAN_IMG=${PCA_ROOT}/mean.nii.gz

OUT_PCA_BIN_PATH=${PCA_ROOT}/model.bin
mkdir -p ${OUT_PC_FOLDER}

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/run_pca_concat_with_mask.py \
  --config ${CONFIG_YAML} \
  --in-ori-folder ${IN_DATA_FOLDER} \
  --in-jac-folder ${IN_JAC_FOLDER} \
  --mask-img-path ${MASK_IMG} \
  --in-file-list ${FILE_LIST} \
  --out-pc-folder ${OUT_PC_FOLDER} \
  --n-components "${N_COMPONENTS}" \
  --batch-size "${BATCH_SIZE}" \
  --save-pca-result-path ${OUT_PCA_BIN_PATH} \
  --load-pca-obj-file-path ${OUT_PCA_BIN_PATH}
set +o xtrace

#set -o xtrace
#${PYTHON_ENV} ${SRC_ROOT}/src/run_pca_concat_with_mask.py \
#  --config ${CONFIG_YAML} \
#  --in-ori-folder ${IN_DATA_FOLDER} \
#  --in-jac-folder ${IN_JAC_FOLDER} \
#  --mask-img-path ${MASK_IMG} \
#  --in-file-list ${FILE_LIST} \
#  --out-pc-folder ${OUT_PC_FOLDER} \
#  --n-components "${N_COMPONENTS}" \
#  --batch-size "${BATCH_SIZE}" \
#  --save-pca-result-path ${OUT_PCA_BIN_PATH}
#set +o xtrace

echo "#####################"
echo "# Step.4 Plot PCs and scree"
ANALYSIS_ROOT=${PROJ_ROOT}/analysis
mkdir -p ${ANALYSIS_ROOT}
PC_FILE_LIST=${PCA_ROOT}/pc_file_list.txt
ls ${OUT_PC_FOLDER} >> ${PC_FILE_LIST}

N_PC_PLOT_PNG=${ANALYSIS_ROOT}/10pc.png
SCREE_PLOT_PNG=${ANALYSIS_ROOT}/scree.png

#set -o xtrace
#${PYTHON_ENV} ${SRC_ROOT}/src/plot_first_n_pc.py \
#  --pc-folder ${OUT_PC_FOLDER} \
#  --file-list-txt ${PC_FILE_LIST} \
#  --out-png ${N_PC_PLOT_PNG}
#set +o xtrace

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/plot_pca_scree.py \
  --load-pca-bin-path ${OUT_PCA_BIN_PATH} \
  --save-img-path ${SCREE_PLOT_PNG}
set +o xtrace





