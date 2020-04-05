#!/bin/bash

# example file. Put this into the project root folder and run.

SRC_ROOT=/nfs/masi/xuk9/singularity/thorax_combine/conda_base/src/thorax_pca
PYTHON_ENV=/home/local/VANDERBILT/xuk9/anaconda3/envs/python37/bin/python

CONFIG=${SRC_ROOT}/config/pca_conda_base.yaml

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DEEDS_OUTPUT_FOLDER=${PROJ_ROOT}/deeds_output
TRANS_FOLDER=${PROJ_ROOT}/trans
JACOBIAN_FOLDER=${PROJ_ROOT}/jacobian
JACOBIAN_TRIM_FOLDER=${PROJ_ROOT}/jacobian_trim
REF_IMG=${PROJ_ROOT}/reference.nii.gz
ROI_IMG=${PROJ_ROOT}/trim_roi_mask.nii.gz
DATA_LIST=${PROJ_ROOT}/file_list

mkdir -p ${TRANS_FOLDER}
mkdir -p ${JACOBIAN_FOLDER}
mkdir -p ${JACOBIAN_TRIM_FOLDER}

echo "Archive output dat files"
for path in ${DEEDS_OUTPUT_FOLDER}/*
do
  scan_name="$(basename -- ${path})"
  ori_trans_file=${path}/non_rigid_displacements.dat
  archived_trans_file=${TRANS_FOLDER}/${scan_name}.nii.gz_displacements.dat
  if [ ! -f "${archived_trans_file}" ]; then
    set -o xtrace
    cp ${ori_trans_file} ${archived_trans_file}
    set +o xtrace
  fi
done

set -o xtrace

${PYTHON_ENV} ${SRC_ROOT}/src/run_get_jacobian.py \
  --config ${CONFIG} \
  --in-folder ${TRANS_FOLDER} \
  --out-folder ${JACOBIAN_FOLDER} \
  --ref-img ${REF_IMG} \
  --data-file-list ${DATA_LIST}

${PYTHON_ENV} ${SRC_ROOT}/src/run_roi_trim.py \
  --config ${CONFIG} \
  --in-folder ${JACOBIAN_FOLDER} \
  --out-folder ${JACOBIAN_TRIM_FOLDER} \
  --roi-img ${ROI_IMG}

set +o xtrace