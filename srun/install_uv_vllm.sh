#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/netscratch/pokarats/tmp/install_done_${SLURM_JOBID}"
# the -P flag works on Linux, but not on macOS
if [[ $SLURM_LOCALID == 0 ]]; then
    # install python-dotenv regardless
    echo "Will install python-dotenv"
    python -m pip install --upgrade pip
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    export UV_LINK_MODE=copy
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
