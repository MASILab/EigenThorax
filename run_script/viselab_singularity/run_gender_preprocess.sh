#!/bin/bash

SRC_ROOT=/src/thorax_pca
PYTHON_ENV=/opt/conda/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_vlsp_singularity.yaml
OUT_ROOT=/data_root/clustering/pca/20200315_vlsp_gender

MALE_FLAG="male"
FEMALE_FLAG="female"
IN_FOLDER=/data_root/affine_vlsp/20200308_vlsp_config_v4/interp/ori

run_preprocess_gender () {
  local GENDER_FLAG=$1

  echo "#####################"
  echo "Preprocess for ${GENDER_FLAG}"
  echo

  local GENDER_FILE_LIST=${OUT_ROOT}/file_list/${GENDER_FLAG}.txt

  local GENDER_ROOT=${OUT_ROOT}/${GENDER_FLAG}
  mkdir -p ${GENDER_ROOT}

  local UNION_AVERAGE_IMG=${GENDER_ROOT}/union_ave.nii.gz
  local OUT_IMPUTE_FOLDER=${GENDER_ROOT}/preprocess/imputation
  mkdir -p ${OUT_IMPUTE_FOLDER}

  echo "#####################"
  echo "# Step.1 Generate union average. We don't need to resample. Use the average atlas as reference."
  set -o xtrace
  ${PYTHON_ENV} ${SRC_ROOT}/src/generate_union_average.py \
    --config ${CONFIG} \
    --in-folder ${IN_FOLDER} \
    --save-path ${UNION_AVERAGE_IMG} \
    --data-file-list ${GENDER_FILE_LIST}
  set +o xtrace
  echo

  echo "#####################"
  echo "# Step.2 Do average imputation on the original scans using the union average."
  set -o xtrace
  ${PYTHON_ENV} ${SRC_ROOT}/src/run_average_imputation.py \
    --config ${CONFIG} \
    --in-folder ${IN_FOLDER} \
    --out-folder ${OUT_IMPUTE_FOLDER} \
    --average-img ${UNION_AVERAGE_IMG} \
    --data-file-list ${GENDER_FILE_LIST}
  set +o xtrace
  echo
}

run_preprocess_gender ${MALE_FLAG}
run_preprocess_gender ${FEMALE_FLAG}
