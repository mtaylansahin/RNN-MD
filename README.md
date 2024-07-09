# Prediction of Protein Interaction Dynamics by Graph Neural Networks

In this study, we aim to examine the temporal dynamics of protein-protein interactions (PPIs) through the application of deep learning techniques. We selected recurrent neural networks (RNNs) for their ability to effectively capture and summarize sequences of PPI dynamics. By modifying the parameters of an existing application ([RE-Net](https://github.com/INK-USC/RE-Net)), we developed a method termed RNN-MD. This approach utilizes historical interaction data from molecular dynamics (MD) simulations and the output from the [interfacea tool](https://github.com/JoaoRodrigues/interfacea) to predict future interactions.

## Code Architecture

Initially, trajectory and topology data are used to generate an MD ensemble, capturing dynamic molecular conformations. The MD ensemble is then analyzed using the Interfacea tool to produce interaction data, including hydrophobic interactions, hydrogen bonds, and salt bridges. This interaction data is split into training, validation, and test sets. A recurrent neural network (RNN) model is pretrained and trained using the training and validation sets to learn patterns from the interaction data. Finally, the trained model is used to predict future interactions, providing valuable insights into molecular behavior.

<p align="center">
<img align="center" src="/code_architecture.png" width = "600" />
</p>


## System Dependencies
* [RE-Net](https://github.com/INK-USC/RE-Net)
* conda 
* pymol


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

## Installation

Clone the repository
```
git clone https://github.com/mehdikosaca/RNN-MD.git
cd RNN-MD
```
After clone the RNN-MD repository, run the following commands for installation requirements:
```
conda create -n renet python=3.6 numpy
conda activate renet
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu101==0.4.3.post2
conda install cudatoolkit=10.1
pip install -r requirements.txt
conda activate renet
```
dgl-cuda10.1<0.5 is not avaiable conda no longer. Instead, move `dgl` and `dgl-0.5.0-py3.6.egg-info` to the environment folder at `.conda/envs/<env_name>/lib/python3.6/site-packages`
This creates a conda environment into the repository and install dependencies to run RNN-MD.

### Quick Installation

Or you can directly run for quick installation of RNN-MD and interfacea under the 1JPS

```
chmod +x setup.sh
./setup.sh
```

## Usage

### Converting an Ensemble PDB File from Trajectory and Topology Data and Generating Interface Outputs

In the first step, the trajectory and topology data need to be converted into an ensemble PDB file. Follow the instructions provided in the README.md file located in the 1JPS folder ([README.md](/1JPS/README.MD)). After completing this step, the interface data must be converted to the [special format](#dataset-formatting) required for processing the RNN-MD for training.

```
python format.py [input_folder] [atomic/residue] [replica_no] [chain 1] [chain 2] [train_ratio] [valid_ratio]
```
This script generates **train.txt, valid.txt, test.txt, and stat.txt** files based on the specified split rule. Move these files to the `RE-Net/data/<case_id>` folder along with the `get_history_graph.py` script and the `labels.txt` file. Then, run the *get_history_graph.py* script to generate history graphs for your training set. Finally, you can train the model with your specified parameters and make predictions by running the scripts below.

```
python model_train.py --dropout [dropout] --learning_rate [learning rate] --batch_size [batch size] --pretrain_epochs [pretrain epochs] --train_epochs [train epochs] --n_hidden [number of hidden] your_file_name_here
python result.py [output_file]
```

If you are working on an HPC and want to run the process using a Slurm file, you should prepare and execute a Slurm file similar to the example below. 

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

In this study, a separate Slurm file has been prepared for hyperparameter optimization due to the numerous parameters tested on RE-Net. If you also wish to perform hyperparameter optimization, you can edit and execute the `hyperparameter_optimization.sh` Slurm file located in the main directory.

## Dataset Formatting
We have used two formats (**atomic and residue**) to represent interactions using `format.py` according to RE-Net input format.
In both descriptions relations were set as the interaction types: hydrophobic, ionic and hydrogen bonding. 
Labels for interaction types were <span style="color:green">0: hbond</span>, <span style="color:blue">1: hydrophobic</span>, <span style="color:red">2: ionic</span>.

### Residue Description
Interacting residues in a dimeric interaction was described by numbering the residues in both chains (with chain B labels being incremented after the last residue number in chain A)

example :
| Subject| relation  | object | time stamp |
| :---:   | :-: | :-: | :---:  |
| 48|	2	|287	|200.0|
.       .    .          .
.       .    .          .
.       .    .          .

### Bug Report & Feedback
If you encounter any problem, you can contact with Ezgi:

### Contacts
- ezgi.karaca@ibg.edu.tr