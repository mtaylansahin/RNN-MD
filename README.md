# Prediction of Protein Interaction Dynamics by Graph Neural Networks

## Motivation

In this study, we aim to examine the temporal dynamics of protein-protein interactions (PPIs) through the application of deep learning technique. We selected recurrent neural networks (RNNs) for their ability to effectively capture and summarize sequences of PPI dynamics. By modifying the parameters of an existing application ([RE-Net](https://github.com/INK-USC/RE-Net)), we developed a method termed RNN-MD. This approach utilizes historical interaction data from molecular dynamics (MD) simulations and the output from the [interfacea tool](https://github.com/JoaoRodrigues/interfacea) to predict future interactions.

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
* seaborn

## Installation

**RNN-MD works on computers with an NVIDIA GPU. Otherwise, you will encounter an error.**

Clone the repository
```
git clone https://github.com/mehdikosaca/RNN-MD.git
cd RNN-MD
```
After clone the RNN-MD repository, run the following commands for installation requirements:
```
conda create -n RNN-MD python=3.6 numpy
conda activate RNN-MD
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu101==0.4.3.post2
conda install cudatoolkit=10.1
pip install -r requirements.txt
conda activate RNN-MD
```

This creates a conda environment into the repository and install dependencies to run RNN-MD.

### Quick Installation

Or you can directly run for quick installation of RNN-MD.

```
chmod +x setup.sh
./setup.sh
```

## Usage

**RNN-MD.py** accepts many paramaters as inputs. You can examine all parameters via `python RNN-MD.py --help` command. If you want to run RNN-MD with default parameters. You must run as following:

```
python RNN-MD.py --data_dir <Interaction_data_dir> --replica <Replica_number> --chain1 <First_chain> --chain2 <Second_chain> --train_ratio <Split_ratio_for_train> --valid_ratio <Split_ratio_for_valid>
```
You can find an example usage of RNN-MD.py following:
```
python RNN-MD.py --data_dir 1JPS --replica 1 --chain1 A --chain2 C --train_ratio 0.8 --valid_ratio 0.1
```
If you want to play with parameters, you should enter them as a string. These parameters can either be a single value or a range formatted as [start, stop, step]. Example usage is given following:
```
python RNN-MD.py --data_dir 1JPS --replica 1 --chain1 A --chain2 C --train_ratio 0.8 --valid_ratio 0.1 --dropout "0.4" --train_epochs "[10, 100, 10]"
```

If you are working on an HPC, you must use a slurm file to run the script. Example file is given below:

```
#!/bin/bash
#SBATCH --partition=ulimited3
#SBATCH --job-name=RNN-MD
#SBATCH --ntasks-per-node=33
#SBATCH --nodes=1
#SBATCH --mail-type=END

module load cuda92/toolkit/9.2.88
source activate RNN-MD

# Run main.py script with the desired arguments
srun python RNN-MD.py --data_dir 1JPS --replica 1 --chain1 A --chain2 C --train_ratio 0.8 --valid_ratio 0.1 --train_epochs "[10, 50, 10]"

exit
```

### Usage of Individual Scripts

* The `format.py` script takes interface outputs as inputs to generates *train.txt*, *valid.txt*, *test.txt*, *stat.txt* files according to you specified train and valid ratios. Example usage is following:

```
python format.py [input_folder] [atomic/residue] [replica_no] [chain 1] [chain 2] [train_ratio] [valid_ratio]
```
- After generating **train.txt, valid.txt, test.txt, and stat.txt**, move them  to the `RE-Net/data/<case_id>` folder along with the `get_history_graph.py` script and the `labels.txt` file. Then, run the *get_history_graph.py* script to generate history graphs for your training set. Finally, you can train the model with your specified parameters, make predictions and create plots by running the scripts below.

```
python model_train.py --dropout [dropout] --learning_rate [learning rate] --batch_size [batch size] --pretrain_epochs [pretrain epochs] --train_epochs [train epochs] --n_hidden [number of hidden] your_file_name_here
python result.py --input_dir [where is inputs (eg.RE-Net/data/1JPS)] --output_dir [where is output move (eg. results/1JPS_results_kg4fsd) --ouput_file_dir (eg. results/1JPS_results_kg4fsd/1JPS_prediction_set_1.txt)]
```

* If you are working on an HPC you must use slurm file to run individual scripts. Example file is given below: 

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

srun python format.py 1JPS residue 1 A C 0.8 0.1
srun python model_train.py --dropout 0.5 --learning_rate 0.001 --batch_size 128 --pretrain_epochs 10 --train_epochs 30 --n_hidden 100 1JPS
srun python result.py --input_dir RE-Net/data/1JPS --output_dir results/1JPS_results_kg4fsd --ouput_file_dir results/1JPS_results_kg4fsd/1JPS_prediction_set_1.txt
 
exit
```

## RNN-MD Output Files

* **Ground Truth List:** The Ground Truth List provides actual values in the dataset. This list serves as the benchmark against which the model's predictions are compared to assess accuracy and performance.

* **Prediction List:** The Prediction List contains the values or outcomes predicted by the RNN-MD model based on the input data. 

* **Performance Metrics:** It provides quantitative evaluations of the RNN-MD model's effectiveness. These metrics typically include measures to accuracy, precision, TPR, FPR, recall, F1 score, and MCC score.

* **Heatmap Similarity Score:** The Heatmap Similarity Score represents a quantitative measure of how similar the interaction patterns predicted by the RNN-MD model are to the actual interaction patterns observed in the ground truth data.

* **All interactions Heatmap:** The All Interaction Heatmap visualizes the all interfacial interactions together with their percentages during MD simulations.

* **Time-dependent interaction plot:** The Time-dependent Interaction Plot illustrates how interactions are observed over time. 

* **Prediction Accuracy Plot:** The Prediction Accuracy Plot visualizes the False Positives and False negatives interactions.

* **Ground Truth vs Prediction Heatmap:** The Ground Truth vs Prediction Heatmap provides a visual comparison between the actual values (ground truth) and the predicted values generated by the model.

* **Ground Truth vs Prediction Bubble Heatmap:** The Ground Truth vs Prediction Bubble Heatmap visualizes the comparison between actual values (ground truth) and predicted values using bubbles.

* **Prediction set in RNN-MD format:** The Prediction Set in RNN-MD Format consists of the predicted values [formatted](#dataset-formatting) specifically for use with the RNN-MD model.

* **Metadata:** The Metadata includes brief information about which RNN-MD parameters are used to generate the predictions.

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