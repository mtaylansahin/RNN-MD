#!/bin/bash

DEST_DIR="/home/$USER/.conda/envs/RNN-MD-2/lib/python3.6/site-packages"


# Create and activate conda environment
conda create -n RNN-MD python=3.6 numpy -y
source activate RNN-MD
# Install PyTorch and dependencies
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu101==0.4.3.post2
mv dgl "$DEST_DIR"
mv dgl-0.5.0-py3.6.egg-info "$DEST_DIR"
 

# Install Python dependencies
pip install -r requirements.txt


# Navigate to interfacea-master directory and set up the environment
cd 1JPS/interfacea-master/

# Add encoding declaration to setup.py
if ! grep -q "^# -*- coding: utf-8 -*-" setup.py; then
    sed -i '1s/^/# -*- coding: utf-8 -*-\n/' setup.py
fi

conda env create -f requirements.yml # Create environment with dependencies

python setup.py build && python setup.py install # Install interfacea

echo "All commands executed successfully."
