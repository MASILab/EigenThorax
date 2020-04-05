#!/bin/bash

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

WRAPPED_FOLDER=${PROJ_ROOT}/wrapped
WRAPPED_TRIM_FOLDER=${PROJ_ROOT}/wrapped_trim
ROI_IMG=${PROJ_ROOT}/trim_roi_mask.nii.gz

set -o xtrace
mkdir -p ${WRAPPED_TRIM_FOLDER}
${PYTHON_ENV} ${SRC_ROOT}/src/run_roi_trim.py \
  --config ${CONFIG} \
  --in-folder ${WRAPPED_FOLDER} \
  --out-folder ${WRAPPED_TRIM_FOLDER} \
  --roi-img ${ROI_IMG}
set +o xtrace