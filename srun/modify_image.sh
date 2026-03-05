#!/bin/bash

srun -K -p RTXA6000-MLT \
  --mem=64GB \
  --time=04:00:00 \
  --immediate=3600 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_25.03-py3.sqsh \
  --container-save=/netscratch/pokarats/"$1".sqsh \
  --pty /bin/bash
