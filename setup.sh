#!/bin/sh
set -e

# Create conda environment
conda create --name fv python=3.8 -y

# Activate the environment
. "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fv

# Install PyTorch and related packages
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install openmim, mmengine, and mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"

# Install mmdetection
pip install -v -e .

# Install mmpretrain
pip install mmpretrain

# Install other dependencies
pip install future tensorboard
pip install ipykernel