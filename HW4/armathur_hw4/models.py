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
            nn.Conv(3, 16, kernel_size=7, stride=4, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(16, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv(16, 32, kernel_size=3, stride=2, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv(32, 32, kernel_size=3, stride=1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv(32, 32, kernel_size=3, stride=1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(32, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv(32, 64, kernel_size=3, stride=2, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(64, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv(64, 128, kernel_size=3, stride=2, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Conv(128, 128, kernel_size=3, stride=1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv(128, 128, kernel_size=3, stride=1, bias=True, device=device, dtype=dtype),
            nn.BatchNorm2d(128, device=device, dtype=dtype),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, bias=True, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, bias=True, device=device, dtype=dtype)

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = self.conv2(x)
        print(f"After conv2: {x.shape}")
        
        identity = x
        x = self.conv3(x)
        print(f"After conv3: {x.shape}")
        x = self.conv4(x)
        print(f"After conv4: {x.shape}")
        x = x + identity
        print(f"After residual1: {x.shape}")
        
        x = self.conv5(x)
        print(f"After conv5: {x.shape}")
        x = self.conv6(x)
        print(f"After conv6: {x.shape}")
        
        identity = x
        x = self.conv7(x)
        print(f"After conv7: {x.shape}")
        x = self.conv8(x)
        print(f"After conv8: {x.shape}")
        x = x + identity
        print(f"After residual2: {x.shape}")
        
        x = self.flatten(x)
        print(f"After flatten: {x.shape}")
        x = self.linear1(x)
        print(f"After linear1: {x.shape}")
        x = self.relu(x)
        print(f"After relu: {x.shape}")
        x = self.linear2(x)
        print(f"After linear2 (output): {x.shape}")
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
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else:
            raise ValueError(f"seq_model must be 'rnn' or 'lstm', got {seq_model}")
        
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
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
