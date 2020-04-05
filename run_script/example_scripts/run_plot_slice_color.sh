#!/bin/bash

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

#CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PNG_OUTPUT_FOLDER=${PROJ_ROOT}/slice_png

mkdir -p ${PNG_OUTPUT_FOLDER}

IN_IMG_FOLDER=jacobian_trim_aff_correct_analyze
#IN_IMG_FOLDER=jacobian_trim_analyze
#IN_IMG_NAME=average.nii.gz
IN_IMG_NAME=variance.nii.gz
IN_IMG=${PROJ_ROOT}/${IN_IMG_FOLDER}/${IN_IMG_NAME}
OUT_PNG=${PNG_OUTPUT_FOLDER}/${IN_IMG_FOLDER}_${IN_IMG_NAME}.png

VMAX="3"
VMIN="-4"

#VMAX="1"
#VMIN="-3"

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/plot_slice_color.py \
  --in-img ${IN_IMG} \
  --out-png ${OUT_PNG} \
  --vmin ${VMIN} \
  --vmax ${VMAX}

set +o xtrace