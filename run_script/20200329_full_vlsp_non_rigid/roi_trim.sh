#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_vlsp.yaml

IN_FOLDER=/nfs/masi/xuk9/SPORE/clustering/registration/non_rigid/full/20200327_full_vlsp_non_rigid/output/non_rigid/wrapped
OUT_FOLDER=/nfs/masi/xuk9/SPORE/clustering/registration/non_rigid/full/20200327_full_vlsp_non_rigid/output/non_rigid/wrapped_roi_trim
ROI_IMG=/nfs/masi/xuk9/SPORE/clustering/registration/non_rigid/full/20200327_full_vlsp_non_rigid/trim/trim_roi_mask.nii.gz

${PYTHON_ENV} ${SRC_ROOT}/src/run_roi_trim.py \
  --config ${CONFIG} \
  --in-folder ${IN_FOLDER} \
  --out-folder ${OUT_FOLDER} \
  --roi-img ${ROI_IMG}