#!/bin/bash

# Find Conda executable path
CONDA_EXE=$(which conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: Conda executable not found. Make sure Conda is installed and accessible in your PATH."
    exit 1
fi

# Determine Conda installation path
CONDA_PATH=$(dirname $(dirname $CONDA_EXE))

# Check if Conda path is found
if [ ! -d "$CONDA_PATH" ]; then
    echo "Error: Conda path '$CONDA_PATH' not found."
    exit 1
fi

echo "Found Conda installation at: $CONDA_PATH"

# Source Conda initialization script
echo "Activating Conda environment..."
. "$CONDA_PATH"/etc/profile.d/conda.sh  # Source Conda script

conda create -n RNN-MD python=3.6 numpy -y

# Activate your specific Conda environment
conda activate RNN-MD  # Replace 'myenv' with your environment name

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment 'RNN-MD'."
    exit 1
fi

echo "Conda environment 'RNN-MD' activated successfully."

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu101==0.4.3.post2
conda install -y cudatoolkit=10.1
pip install -r requirements.txt