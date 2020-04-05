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
STD_ROI_MASK=${PROJ_ROOT}/reference/roi_mask.nii.gz
IMG_MASK_OVERLAY_PNG_FOLDER=${ANALYSIS_ROOT}/affine_img_mask_overlay_png

mkdir -p ${IMG_MASK_OVERLAY_PNG_FOLDER}

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/plot_img_mask_overlay.py \
  --config ${CONFIG} \
  --in-folder ${AFFINE_INTERP_FOLDRE} \
  --mask-img ${STD_ROI_MASK} \
  --out-png-folder ${IMG_MASK_OVERLAY_PNG_FOLDER} \
  --file-list-txt ${DATA_LIST}

set +o xtrace