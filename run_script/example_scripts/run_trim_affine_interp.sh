#!/bin/bash

# ln -s /nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca/run_script/example_scripts/${script_name} ./script_name

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/opt/conda/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

AFFINE_ROOT=${PROJ_ROOT}/output/affine
#NON_RIGID_ROOT=${PROJ_ROOT}/output/non_rigid
ANALYSIS_ROOT=${PROJ_ROOT}/analysis

mkdir -p ${ANALYSIS_ROOT}

#DEEDS_OUTPUT_FOLDER=${NON_RIGID_ROOT}/deeds_output
#TRANS_FOLDER=${NON_RIGID_ROOT}/trans
#JACOBIAN_FOLDER=${ANALYSIS_ROOT}/jacobian
#JACOBIAN_TRIM_FOLDER=${ANALYSIS_ROOT}/jacobian_trim
#REF_IMG=${PROJ_ROOT}/reference/roi_mask.nii.gz

AFFINE_INTERP_FOLDER=${AFFINE_ROOT}/interp/ori
AFFINE_INTERP_TRIM_FOLDER=${ANALYSIS_ROOT}/affine_interp_trim

ROI_IMG=${PROJ_ROOT}/reference/trim_roi_mask.nii.gz
DATA_LIST=${PROJ_ROOT}/data/file_list

mkdir -p ${AFFINE_INTERP_TRIM_FOLDER}

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/run_roi_trim.py \
  --config ${CONFIG} \
  --in-folder ${AFFINE_INTERP_FOLDER} \
  --out-folder ${AFFINE_INTERP_TRIM_FOLDER} \
  --roi-img ${ROI_IMG}

set +o xtrace