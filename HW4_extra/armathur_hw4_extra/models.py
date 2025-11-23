import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = nn.Sequential(
            nn.Conv(3, 16, kernel_size=7, stride=4, padding=3, bias=False, device=device, dtype=dtype),
            nn.BatchNorm2d(16, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(16, 32, kernel_size=3, stride=2, padding=1, bias=False, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv(32, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(32, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv(32, 64, kernel_size=3, stride=2, padding=1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(64, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.downsample = nn.Conv(32, 128, kernel_size=1, stride=2, bias=False, device=device, dtype=dtype)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*2*2, 592, device=device, dtype=dtype)
        self.linear2 = nn.Linear(592, 10, device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x)

        identity = x
        x = self.conv2(x)
        x = x + identity

        identity = self.downsample(x)
        x = self.conv3(x)
        x = x + identity

        x = self.flatten(x)
        x = nn.ReLU()(self.linear(x))
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model_type = seq_model
        
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
            self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        elif seq_model == 'transformer':
            self.seq_model = nn.Transformer(embedding_size, hidden_size, num_layers, 
                                           device=device, dtype=dtype, 
                                           sequence_len=seq_len)
            self.linear = nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        else:
            raise ValueError(f"seq_model must be 'rnn', 'lstm', or 'transformer', got {seq_model}")
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        
        x = self.embedding(x)
        
        x, h = self.seq_model(x, h)
        
        if self.seq_model_type == 'transformer':
            x = x.reshape((seq_len * bs, self.embedding_size))
        else:
            x = x.reshape((seq_len * bs, self.hidden_size))
        
        out = self.linear(x)
        
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
