#!/bin/bash

# The entry point for singularity call. Put this in project root folder.
# Need to add this folder to /.singularity.d/env/95-apps.sh

# e.g. _pc_run_affine_pipeline.sh
RUN_COMMAND="$(basename -- $1)"

PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# /data_root should be bond to the root location of the registration pipeline.
DATA_ROOT=${PROJ_ROOT}/../

SINGULARITY_PATH=${MY_NFS_DATA_ROOT}/singularity/thorax_combine/conda_base

set -o xtrace
singularity exec \
            -B ${PROJ_ROOT}:/proj_root \
            -B ${DATA_ROOT}:/data_root \
            ${SINGULARITY_PATH} ${RUN_COMMAND}
set +o xtrace
