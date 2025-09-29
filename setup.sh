#!/bin/sh
set -e

# Create conda environment
conda create --name fv python=3.8 -y

# Activate the environment
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fv

# Install CUDA toolkit
conda install cudatoolkit=11.7 -c nvidia -y

# Install PyTorch and related packages
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install openmim, mmengine, and mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"

# Install mmdetection
pip install -v -e .