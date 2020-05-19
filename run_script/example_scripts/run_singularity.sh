#!/bin/bash

# This is an example file. Put it into the project root.

RUN_COMMAND="$(basename -- $1)"

########### need set manually #############
DATA_ROOT=
PROJ_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# THORAX_SINGULARITY_IMG is defined as enviroment variable.
SINGULARITY_PATH=${THORAX_SINGULARITY_IMG}

set -o xtrace
singularity exec \
            -B ${PROJ_ROOT}:/proj_root \
            -B ${DATA_ROOT}:/data_root \
            ${SINGULARITY_PATH} ${RUN_COMMAND}
set +o xtrace
