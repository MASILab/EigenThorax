#!/bin/bash

# example file. Put this into the project root folder and run.
# ln -s /nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca/run_script/example_scripts/${script_name} ./script_name

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/opt/conda/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

AFFINE_ROOT=${PROJ_ROOT}/output/affine
NON_RIGID_ROOT=${PROJ_ROOT}/output/non_rigid
ANALYSIS_ROOT=${PROJ_ROOT}/analysis

DATA_LIST=${PROJ_ROOT}/data/file_list
NON_RIGID_WRAPPED_FOLDER=${NON_RIGID_ROOT}/wrapped
NON_RIGID_REGION_MASK_FOLDER=${NON_RIGID_ROOT}/preprocess/reg_roi_mask
AVERAGE_W_REGION_MASK_IMG=${ANALYSIS_ROOT}/average_w_region_mask.nii.gz

mkdir -p ${ANALYSIS_ROOT}

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/run_get_average_with_region_mask.py \
  --config ${CONFIG} \
  --in-scan-folder ${NON_RIGID_WRAPPED_FOLDER} \
  --in-mask-folder ${NON_RIGID_REGION_MASK_FOLDER} \
  --out-average-img ${AVERAGE_W_REGION_MASK_IMG} \
  --file-list-txt ${DATA_LIST}

set +o xtrace