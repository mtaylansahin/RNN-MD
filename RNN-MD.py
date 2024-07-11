import argparse
import os
import shutil
import subprocess
import sys
import numpy as np
import random
import string


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script for predicting protein-protein interaction dynamics.')

    # Add arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the interaction files.')
    parser.add_argument('--replica', type=str, required=True, help='Replica number.')
    parser.add_argument('--chain1', type=str, required=True, help='First chain.')
    parser.add_argument('--chain2', type=str, required=True, help='Second chain.')
    parser.add_argument('--train_ratio', type=str, required=True, help='Training ratio for splitting the data.')
    parser.add_argument('--valid_ratio', type=str, required=True, help='Validation ratio for splitting the data.')
    parser.add_argument('--dropout', type=str, default="0.5", help='Dropout rate for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--learning_rate', type=str, default="0.001", help='Learning rate for the optimizer. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--batch_size', type=str, default="128", help='Batch size for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--pretrain_epochs', type=str, default="30", help='Number of epochs for pre-training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--train_epochs', type=str, default="10", help='Number of epochs for training. Can be a single value or a range in the format [start, stop, step].')
    parser.add_argument('--n_hidden', default="200", type=str, help='Number of hidden units in the hidden layer. Can be a single value or a range in the format [start, stop, step].')

    # Parse arguments
    args = parser.parse_args()

    return args

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

def copy_txt_files(src_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    # Iterate over all files in the source directory
    for item in os.listdir(src_dir):
        # Construct full file path
        source_file = os.path.join(src_dir, item)
        # If it's a file and ends with .txt, copy it to the destination directory
        if os.path.isfile(source_file) and item.endswith(".txt"):
            shutil.copy2(source_file, dest_dir)
            print(f"Copied {source_file} to {dest_dir}")
        # If it's the get_history_graph.py script, copy it to the destination directory
        elif os.path.isfile(source_file) and item == "get_history_graph.py":
            shutil.copy2(source_file, dest_dir)
            print(f"Copied {source_file} to {dest_dir}")

def generate_hex_id(length=5):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class ModelTrainTest:

    def __init__(self, file_name):
        self.file_name = file_name

    def get_hist_graph(self):
        current_dir = os.getcwd()
        try:
            os.chdir(os.path.join(current_dir, 'RE-Net/data', self.file_name))
            os.system("python get_history_graph.py")
        finally:
            os.chdir(current_dir)

    def pretrain(self, dropout, n_hidden, learning_rate, pretrain_epochs, batch_size):
        current_dir = os.getcwd()
        try:
            os.chdir(os.path.join(current_dir, 'RE-Net'))
            os.system(f"python pretrain.py -d {self.file_name} --gpu 0 --dropout {dropout} --n-hidden {n_hidden} --lr {learning_rate} --max-epochs {pretrain_epochs} --batch-size {batch_size}")
        finally:
            os.chdir(current_dir)

    def train(self, dropout, n_hidden, learning_rate, train_epochs, batch_size):
        current_dir = os.getcwd()
        try:
            os.chdir(os.path.join(current_dir, 'RE-Net'))
            os.system(f"python train.py -d {self.file_name} --gpu 0 --dropout {dropout} --n-hidden {n_hidden} --lr {learning_rate} --max-epochs {train_epochs} --batch-size {batch_size} --num-k 5")
        finally:
            os.chdir(current_dir)

    def test(self, n_hidden, dropout, learning_rate, train_epochs, batch_size, run_id, results_dir):
        current_dir = os.getcwd()
        try:
            os.chdir(os.path.join(current_dir, 'RE-Net'))
            os.system(f"python test.py -d {self.file_name} --gpu 0 --n-hidden {n_hidden} --num-k 5")
            new_output_file = os.path.join(results_dir, f"{self.file_name}_prediction_set_{run_id}.txt")
            # Create metadata file in results directory
            metadata_file = os.path.join(results_dir, f"{self.file_name}_metadata_{run_id}.txt")
            os.makedirs(results_dir, exist_ok=True)
            with open(metadata_file, "w") as f:
                f.write(f"Parameters for {new_output_file}:\n")
                f.write(f"data_dir: {self.file_name}\n")
                f.write(f"dropout: {dropout}\n")
                f.write(f"n_hidden: {n_hidden}\n")
                f.write(f"learning_rate: {learning_rate}\n")
                f.write(f"train_epochs: {train_epochs}\n")
                f.write(f"batch_size: {batch_size}\n")
            # Move metadata file to the results directory
            shutil.move(f"{self.file_name}_prediction_set_{run_id}.txt", results_dir)
        finally:
            os.chdir(current_dir)

if __name__ == "__main__":
    dir = os.getcwd()

    args = parse_arguments()
    
    dropout_values = parse_range(args.dropout, float)
    learning_rate_values = parse_range(args.learning_rate, float)
    batch_size_values = parse_range(args.batch_size, int)
    pretrain_epochs_values = parse_range(args.pretrain_epochs, int)
    train_epochs_values = parse_range(args.train_epochs, int)
    n_hidden_values = parse_range(args.n_hidden, int) if args.n_hidden else None

    print(f"Data directory: {args.data_dir}")
    print(f"Replica number: {args.replica}")
    print(f"First chain: {args.chain1}")
    print(f"Second chain: {args.chain2}")
    print(f"Training ratio: {args.train_ratio}")
    print(f"Validation ratio: {args.valid_ratio}")
    print(f"Dropout rates: {dropout_values}")
    print(f"Learning rates: {learning_rate_values}")
    print(f"Batch sizes: {batch_size_values}")
    print(f"Number of pre-training epochs: {pretrain_epochs_values}")
    print(f"Number of training epochs: {train_epochs_values}")
    print(f"Number of hidden units: {n_hidden_values}")

    # Create data_dir inside RE-Net/data/
    data_dir_path = os.path.join("RE-Net", "data", args.data_dir)
    os.makedirs(data_dir_path, exist_ok=True)
    print(f"Directory '{data_dir_path}' created (or already exists).")

    # Copy all .txt files and the get_history_graph.py script from the input data_dir to the new data_dir
    copy_txt_files(args.data_dir, data_dir_path)

    # Call the format.py script with the specified arguments
    format_script_path = "format.py"  # Adjust the path as needed
    format_args = [
        args.data_dir,
        "residue",  # or "atomic", adjust based on your use case
        args.replica,
        args.chain1,
        args.chain2,
        args.train_ratio,
        args.valid_ratio
    ]

    try:
        result = subprocess.run([sys.executable, format_script_path] + format_args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        print(result.stderr.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running format.py: {e}")
        print(e.stdout.decode('utf-8'))
        print(e.stderr.decode('utf-8'))

    # Create the base results directory
    results_base_dir = "../results"
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"Directory '{results_base_dir}' created (or already exists).")

    # Model training and testing
    model_trainer = ModelTrainTest(args.data_dir)  # Using data_dir as file_name
    model_trainer.get_hist_graph()
    
    run_id = 1
    for dropout in dropout_values:
        for n_hidden in n_hidden_values:
            for learning_rate in learning_rate_values:
                for pretrain_epoch in pretrain_epochs_values:
                    for batch_size in batch_size_values:
                        model_trainer.pretrain(dropout, n_hidden, learning_rate, pretrain_epoch, batch_size)
                        for train_epoch in train_epochs_values:
                            model_trainer.train(dropout, n_hidden, learning_rate, train_epoch, batch_size)
                            hex_id = generate_hex_id()
                            results_dir = os.path.join(results_base_dir, f"{args.data_dir}_results_{hex_id}")
                            os.makedirs(results_dir, exist_ok=True)
                            model_trainer.test(n_hidden, dropout, learning_rate, train_epoch, batch_size, run_id, results_dir)
                            results_dir_actual  = os.path.join("results", f"{args.data_dir}_results_{hex_id}")
                            # Call the result script's main function
                            os.system(f"python result.py --input_dir {data_dir_path} --output_dir {results_dir_actual} --output_file_dir {os.path.join(results_dir_actual, f'{args.data_dir}_prediction_set_{run_id}.txt')}")
                            run_id += 1
