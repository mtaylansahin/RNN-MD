# Prediction of Protein Interaction Dynamics by Graph Neural Networks

Molecular dynamics simulations are an integral process for understanding protein-protein interactions (PPIs). However, they can be computationally expensive and time-consuming. In this study we explore whether machine learning can be integrated with molecular dynamics to reduce the costs of the simulations. For this we have selected and trained the [RE-Net](https://github.com/INK-USC/RE-Net) model from [pytorch geometric library](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html) and analyzed the predicted interactions.

## System Dependencies
* [RE-Net](https://github.com/INK-USC/RE-Net)
* conda 


## Python Dependencies
* numpy
* pandas
* sklearn
* torch
* cudatoolkit
* dglteam
* matplotlib
* plotly
* kaleido

## Installing
After clone the RE-Net, run the following commands:
```
conda create -n renet python=3.6 numpy
conda activate renet
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c dglteam "dgl-cuda10.1<0.5"
conda install cudatoolkit=10.1
pip install -r requirements.txt
pip install -U kaleido
```
This creates a conda environment into the repository and install dependencies to run MD_ML.

## Usage
Conda environment must be activated before use the RNN-MD.
```
conda activate renet
```
And, run the individual scripts in the specified order.
```
python format.py [input_folder] [atomic/residue] [replica_no] [chain 1] [chain 2] [train_ration] [valid_ratio]
```
Before using the `model_train.py`, please ensure the `get_history_graph.py` script under the `/RE-Net/data/<your_case>`directory. Then you can use it as following:
```
python model_train.py --dropout [dropout] --learning_rate [learning rate] --batch_size [batch size] --pretrain_epochs [pretrain epochs] --train_epochs [train epochs] --n_hidden [number of hidden] your_file_name_here
python result.py [output_file]
```

Please make sure load the cuda if you are running on HPC.
```
module load [cuda_toolkit]
```
It is an example slurm content for HPC:
```
#!/bin/bash
#SBATCH --partition=ulimited3
#SBATCH --job-name=RNN-MD
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=[mail]
#SBATCH --output=md_ml_%j.out
#SBATCH --error=md_ml_%j.err


module load cuda92/toolkit/9.2.88

srun python format.py 1JPS residue 1 A C 0.8 0.9
srun python model_train.py --dropout 0.5 --learning_rate 0.001 --batch_size 128 --pretrain_epochs 10 --train_epochs 30 --n_hidden 100 1JPS
srun python result.py output_renet_1JPS_default.txt
 
exit
```

If you want to apply hyperparameter optimization to the case, you can run `hyperparameter_optimization.sh` slurm file at your HPC.

## Dataset Formatting
We have used two formats to represent interactions using `format.py` according to RE-Net input format.
In both descriptions relations were set as the interaction types: hydrophobic, ionic and hydrogen bonding. 
Labels for interaction types were <span style="color:green">0: hbond</span>, <span style="color:blue">1: hydrophobic</span>, <span style="color:red">2: ionic</span>.

### 1- Residue Description
Interacting residues in a dimeric interaction was described by numbering the residues in both chains (with chain B labels being incremented after the last residue number in chain A)

example :
| Subject| relation  | object | time stamp |
| :---:   | :-: | :-: | :---:  |
| 48|	2	|287	|200.0|
.       .    .          .
.       .    .          .
.       .    .          .

