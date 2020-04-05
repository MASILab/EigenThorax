#!/bin/bash

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT=/nfs/masi/xuk9/SPORE/clustering/registration/non_rigid/full/20200327_full_vlsp_non_rigid

IN_FOLDER=${PROJ_ROOT}/output/non_rigid/trans
OUT_FOLDER=${PROJ_ROOT}/output/non_rigid/jacobian
REF_IMG=${PROJ_ROOT}/reference/non_rigid.nii.gz
DATA_LIST=${PROJ_ROOT}/data/file_list

mkdir -p ${OUT_FOLDER}

${PYTHON_ENV} ${SRC_ROOT}/src/run_get_jacobian.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --out-folder ${OUT_FOLDER} \
  --ref-img ${REF_IMG} \
  --data-file-list ${DATA_LIST}