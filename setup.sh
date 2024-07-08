#!/bin/bash

# Create and activate conda environment
conda create -n renet python=3.6 numpy -y

# Install PyTorch and dependencies
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c dglteam dgl-cuda10.1==0.5* -y
conda install cudatoolkit=10.1 -y

# Install Python dependencies
pip install -r requirements.txt

# Install PyMOL
sudo apt-get install pymol -y

# Navigate to interfacea-master directory and set up the environment
cd 1JPS/interfacea-master/
conda env create -f requirements.yml # Create environment with dependencies
python setup.py build && python setup.py install # Install interfacea

echo "All commands executed successfully."
