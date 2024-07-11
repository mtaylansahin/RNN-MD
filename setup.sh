#!/bin/bash

DEST_DIR="/home/$USER/.conda/envs/RNN-MD-2/lib/python3.6/site-packages"


# Create and activate conda environment
conda create -n RNN-MD python=3.6 numpy -y
source activate RNN-MD
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu101==0.4.3.post2
pip install -r requirements.txt

