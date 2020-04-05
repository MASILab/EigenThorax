#!/bin/bash

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

AFFINE_MAT_FOLDER=${PROJ_ROOT}/../affine/omat
JACOBIAN_TRIM_FOLDER=${PROJ_ROOT}/jacobian_trim
JACOBIAN_TRIM_AFFINE_CORRECT_FOLDER=${PROJ_ROOT}/jacobian_trim_aff_correct
REF_IMG=${PROJ_ROOT}/reference.nii.gz
DATA_LIST=${PROJ_ROOT}/file_list

mkdir -p ${JACOBIAN_TRIM_AFFINE_CORRECT_FOLDER}

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/run_jacobian_affine_correct.py \
  --config ${CONFIG} \
  --in-folder ${JACOBIAN_TRIM_FOLDER} \
  --in-affine-folder ${AFFINE_MAT_FOLDER} \
  --out-folder ${JACOBIAN_TRIM_AFFINE_CORRECT_FOLDER} \
  --ref-img ${REF_IMG} \
  --data-file-list ${DATA_LIST}

set +o xtrace