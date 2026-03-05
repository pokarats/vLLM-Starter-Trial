#!/bin/bash

# SLURM environment arguments
IMAGE=/netscratch/pokarats/qwen_uv_vllm_pytorch_25.03-py3_cuda128.sqsh
NUM_CPUS=6
MEM="$2"GB

# variables for srun and python
srun -K -p $1 \
	--container-mounts=/netscratch/pokarats:/netscratch/pokarats,/ds:/ds:ro,"$(pwd)":"$(pwd)" \
        --container-workdir="$(pwd)" \
	      --export=ALL,HF_HUB_CACHE=/netscratch/pokarats/models/llms/cache \
        --container-image=$IMAGE \
        --job-name=test_vllm_"$3" \
        --cpus-per-task=$NUM_CPUS \
        --gpus=1 \
        --mem=$MEM \
        --nodes=1 \
        --mail-type=END,FAIL \
        --mail-user=noon.pokaratsiri@dfki.de \
srun/install_uv_vllm.sh python src/offline_visionExample.py
