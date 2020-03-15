#!/bin/bash

SRC_ROOT=/home/local/VANDERBILT/xuk9/03-Projects/05-ThoraxPCA
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

OUT_ROOT=/nfs/masi/xuk9/SPORE/clustering/pca/2020313_vlsp
PCA_ROOT=${OUT_ROOT}/pca_3
OUT_PCA_BIN_PATH=${PCA_ROOT}/model.bin
OUT_SCREE_PATH=${PCA_ROOT}/scree.png

set -o xtrace
${PYTHON_ENV} ${SRC_ROOT}/src/plot_pca_scree.py \
  --load-pca-bin-path ${OUT_PCA_BIN_PATH} \
  --save-img-path ${OUT_SCREE_PATH}
set +o xtrace
