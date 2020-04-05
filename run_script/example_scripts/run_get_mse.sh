#!/bin/bash

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/opt/conda/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

JACOBIAN_TRIM_FOLDER=${PROJ_ROOT}/jacobian_trim
JACOBIAN_TRIM_AVERAGE=${JACOBIAN_TRIM_FOLDER}_analyze/average.nii.gz
JACOBIAN_TRIM_AFFINE_CORRECT_FOLDER=${PROJ_ROOT}/jacobian_trim_aff_correct
JACOBIAN_TRIM_AFFINE_CORRECT_AVERAGE=${JACOBIAN_TRIM_AFFINE_CORRECT_FOLDER}_analyze/average.nii.gz
DATA_LIST=${PROJ_ROOT}/file_list

OUT_CSV_FOLDER=${PROJ_ROOT}/out_csv
mkdir -p ${OUT_CSV_FOLDER}
MSE_CSV=${OUT_CSV_FOLDER}/mse.csv
MSE_W_AFFINE_CORC_CSV=${OUT_CSV_FOLDER}/mse_w_affine.csv

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/run_get_mse.py \
  --config ${CONFIG} \
  --in-folder ${JACOBIAN_TRIM_FOLDER} \
  --ref-img ${JACOBIAN_TRIM_AVERAGE} \
  --out-csv ${MSE_CSV} \
  --data-file-list ${DATA_LIST}

${PYTHON_ENV} ${SRC_ROOT}/src/run_get_mse.py \
  --config ${CONFIG} \
  --in-folder ${JACOBIAN_TRIM_AFFINE_CORRECT_FOLDER} \
  --ref-img ${JACOBIAN_TRIM_AFFINE_CORRECT_AVERAGE} \
  --out-csv ${MSE_W_AFFINE_CORC_CSV} \
  --data-file-list ${DATA_LIST}

set +o xtrace