#!/bin/bash

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

AFFINE_ROOT=${PROJ_ROOT}/output/affine
ANALYSIS_ROOT=${PROJ_ROOT}/analysis

DATA_LIST=${PROJ_ROOT}/data/file_list
AFFINE_INTERP_FOLDRE=${AFFINE_ROOT}/interp/ori

NON_NAN_MASK_FOLDER=${ANALYSIS_ROOT}/non_nan_mask

mkdir -p ${NON_NAN_MASK_FOLDER}

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/run_get_non_nan_region.py \
  --config ${CONFIG} \
  --in-folder ${AFFINE_INTERP_FOLDRE} \
  --out-folder ${NON_NAN_MASK_FOLDER} \
  --file-list-txt ${DATA_LIST}

set +o xtrace