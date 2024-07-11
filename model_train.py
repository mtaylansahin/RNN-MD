import os
import argparse
import numpy as np
import subprocess

class ModelTrainTest:

    def __init__(self, file_name):
        self.file_name = file_name

    def get_hist_graph(self):
        os.chdir(os.path.join(os.getcwd(), 'RE-Net/data', self.file_name))
        os.system("python get_history_graph.py")

    def pretrain(self, dropout, n_hidden, lr, pretrain_epochs, batch_size):
        os.chdir('../..')
        command = f"python3 pretrain.py -d {self.file_name} --gpu 0 --dropout {dropout} --n-hidden {n_hidden} --lr {lr} --max-epochs {pretrain_epochs} --batch-size {batch_size}"
        self.run_command(command)

    def train(self, dropout, n_hidden, lr, train_epochs, batch_size):
        command = f"python3 train.py -d {self.file_name} --gpu 0 --dropout {dropout} --n-hidden {n_hidden} --lr {lr} --max-epochs {train_epochs} --batch-size {batch_size} --num-k 5"
        self.run_command(command)

    def test(self, n_hidden, dropout, lr, train_epochs, batch_size, run_id):
        command = f"python3 test.py -d {self.file_name} --gpu 0 --n-hidden {n_hidden} --num-k 5"
        self.run_command(command)
        # Rename the output file
        new_output_file = f"{self.file_name}_prediction_set_{run_id}.txt"
        # Create metadata file
        metadata_file = f"{self.file_name}_metadata_{run_id}.txt"
        with open(metadata_file, "w") as f:
            f.write(f"Parameters for {new_output_file}:\n")
            f.write(f"data_dir: {self.file_name}\n")
            f.write(f"dropout: {dropout}\n")
            f.write(f"n_hidden: {n_hidden}\n")
            f.write(f"lr: {lr}\n")
            f.write(f"train_epochs: {train_epochs}\n")
            f.write(f"batch_size: {batch_size}\n")

    def run_command(self, command):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            print(f"Standard output: {stdout.decode('utf-8')}")
            print(f"Standard error: {stderr.decode('utf-8')}")
        else:
            print(f"Command succeeded with return code {process.returncode}")
            print(f"Standard output: {stdout.decode('utf-8')}")

def parse_range(param, param_type=float):
    if param is None:
        return None
    try:
        if param.startswith('[') and param.endswith(']'):
            values = list(map(param_type, param[1:-1].split(',')))
            if len(values) == 3:
                start, stop, step = values
                return list(np.arange(start, stop, step))
            else:
                return values
        else:
            return [param_type(param)]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Parameter {param} is not a valid range or single value.")

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--dropout', type=str, default="0.5", help='Dropout rate for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--learning_rate', type=str, default="0.001", help='Learning rate for the optimizer. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--batch_size', type=str, default="128", help='Batch size for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--pretrain_epochs', type=str, default="30", help='Number of epochs for pre-training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--train_epochs', type=str, default="10", help='Number of epochs for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--n_hidden', type=str, default="100", help='Number of hidden layers for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('file_name', type=str, help='File name for training data')
    
    args = parser.parse_args()

    s = ModelTrainTest(args.file_name)

    dropout_values = parse_range(args.dropout, float)
    learning_rate_values = parse_range(args.learning_rate, float)
    batch_size_values = parse_range(args.batch_size, int)
    pretrain_epochs_values = parse_range(args.pretrain_epochs, int)
    train_epochs_values = parse_range(args.train_epochs, int)
    n_hidden_values = parse_range(args.n_hidden, int)

    s.get_hist_graph()
    
    # Iterate over all combinations of hyperparameters and run training/testing
    run_id = 1
    for dropout in dropout_values:
        for n_hidden in n_hidden_values:
            for lr in learning_rate_values:
                for pretrain_epoch in pretrain_epochs_values:
                    for batch_size in batch_size_values:
                        s.pretrain(dropout, n_hidden, lr, pretrain_epoch, batch_size)
                        for train_epoch in train_epochs_values:
                            s.train(dropout, n_hidden, lr, train_epoch, batch_size)
                            s.test(n_hidden, dropout, lr, train_epoch, batch_size, run_id)
                            run_id += 1

if __name__ == "__main__":
    main()
