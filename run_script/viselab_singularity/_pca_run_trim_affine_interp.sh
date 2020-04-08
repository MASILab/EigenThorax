#!/bin/bash

# ln -s /nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca/run_script/example_scripts/${script_name} ./script_name

# example file. Put this into the project root folder and run.

SRC_ROOT=/src/thorax_pca
PYTHON_ENV=/opt/conda/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_vlsp_singularity.yaml

PROJ_ROOT=/proj_root

AFFINE_ROOT=${PROJ_ROOT}/output/affine
ANALYSIS_ROOT=${PROJ_ROOT}/analysis

mkdir -p ${ANALYSIS_ROOT}

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