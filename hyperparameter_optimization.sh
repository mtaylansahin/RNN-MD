#!/bin/bash
#SBATCH --partition=unlimited3
#SBATCH --job-name=RNN-MD_$CASE
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=[mail]
#SBATCH --output=md_ml_%j.out
#SBATCH --error=md_ml_%j.err

read -p "Enter the path to the data directory: " CASE_DIR
read -p "Enter your file name to analyze: " CASE
cd $CASE_DIR

module load cuda9.2/toolkit/9.2.88

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.1 --n-hidden 100 --lr 1e-3 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.1 --n-hidden 100 --lr 1e-3 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_dropout_0.1.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.2 --n-hidden 100 --lr 1e-3 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.2 --n-hidden 100 --lr 1e-3 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_dropout_0.2.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 200 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_n_hidden_200.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 300 --lr 1e-3 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 300 --lr 1e-3 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 300 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_n_hidden_300.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 400 --lr 1e-3 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 400 --lr 1e-3 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 400 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_n_hidden_400.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-5 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-5 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_lr_1e-5.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-4 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-4 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_lr_1e-4.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_default.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-2 --max-epochs 30 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-2 --max-epoch 10 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_lr_1e-2.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epochs 25 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epoch 25 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_max_epoch_25.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epochs 50 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epoch 50 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_max_epoch_50.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epochs 100 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epoch 100 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_max_epoch_100.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epochs 150 --batch-size 128
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epoch 150 --batch-size 128 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_max_epoch_150.txt

srun python pretrain.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epochs 30 --batch-size 64
srun python train.py -d $CASE --gpu 0 --dropout 0.5 --n-hidden 100 --lr 1e-3 --max-epoch 10 --batch-size 64 --num-k 5
srun python test.py -d $CASE --gpu 0 --n-hidden 100 --num-k 5
mv output_renet_"$CASE".txt output_renet_"$CASE"_batch_size_64.txt

exit
