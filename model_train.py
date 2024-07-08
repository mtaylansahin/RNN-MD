import os
import argparse

class ModelTrainTest:

    def __init__(self, file_name):
        self.file_name = file_name

    def get_hist_graph(self):
        os.chdir(os.path.join(os.getcwd(), 'RE-Net/data', self.file_name))
        os.system("python get_history_graph.py")

    def pretrain(self, dropout, n_hidden, learning_rate, pretrain_epochs, batch_size):
        os.chdir('../..')
        os.system(f"python3 pretrain.py -d {self.file_name} --gpu 0 --dropout {dropout} --n-hidden {n_hidden} --learning_rate {learning_rate} --pretrain_epochs {pretrain_epochs} --batch_size {batch_size}")

    def train(self, dropout, n_hidden, learning_rate, train_epochs, batch_size):
        os.system(f"python3 train.py -d {self.file_name} --gpu 0 --dropout {dropout} --n-hidden {n_hidden} --learning_rate {learning_rate} --train_epochs {train_epochs} --batch_size {batch_size} --num-k 5")

    def test(self, n_hidden):
        os.system(f"python3 test.py -d {self.file_name} --gpu 0 --n-hidden {n_hidden} --num-k 5")

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--pretrain_epochs', type=int, default=30, help='Number of epochs for pre-training')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--n_hidden', type=int, default=100, help='Number of hidden layers for training')
    parser.add_argument('file_name', type=str, help='File name for training data')
    
    args = parser.parse_args()

    s = ModelTrainTest(args.file_name)

    dropout = args.dropout
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    pretrain_epochs = args.pretrain_epochs
    train_epochs = args.train_epochs
    n_hidden = args.n_hidden

    s.get_hist_graph()
    s.pretrain(dropout, n_hidden, learning_rate, pretrain_epochs, batch_size)
    s.train(dropout, n_hidden, learning_rate, train_epochs, batch_size)
    s.test(n_hidden)

if __name__ == "__main__":
    main()
