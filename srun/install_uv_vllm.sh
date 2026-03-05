#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/netscratch/pokarats/tmp/install_done_${SLURM_JOBID}"
# the -P flag works on Linux, but not on macOS
PYVERSION="$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')"
if [[ $SLURM_LOCALID == 0 ]]; then
    if [[ $PYVERSION < 3.8 ]]; then
      # only install this for python version older than 3.8
      echo "$PYVERSION needs to install pickle5"
      pip install pickle5
    else
      echo "$PYVERSION no need to install pickle5, already supported"
    fi
    # install python-dotenv regardless
    echo "Will install python-dotenv"
    python -m pip install --upgrade pip
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    uv venv
    source .venv/bin/activate
    uv pip install -U vllm
    uv pip install qwen-vl-utils==0.0.14
    #pip install python-dotenv
    # Tell other tasks we are done installing
    touch "${DONEFILE}"
else
# Wait until packages are installed
    while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
# This runs your wrapped command
"$@"
