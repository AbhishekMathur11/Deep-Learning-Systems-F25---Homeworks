import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    # ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    layers = []

    layers.append(nn.Linear(dim, hidden_dim))
    layers.append(nn.ReLU())

    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob))

    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    total_loss = 0.0
    total_error = 0.0
    total_samples = 0

    if opt is not None:
        model.train()
    else:
        model.eval()

    loss_fn = nn.SoftmaxLoss()
    for batch in dataloader:
        if len(batch) == 2:
            X, y = batch

        else:
            X = batch[0]
            continue

        X = X.reshape((X.shape[0], -1))

        logits = model(X)
        loss = loss_fn(logits, y)

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        
        batch_size = X.shape[0]
        total_loss += loss.numpy()*batch_size

        predictions = np.argmax(logits.numpy(), axis=1)
        errors = np.sum(predictions != y.numpy())
        total_error += errors
        total_samples += batch_size
    avg_loss = total_loss / total_samples
    error_rate = total_error / total_samples

    return error_rate, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    # ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz",
    )

    train_dataloader = ndl.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model = MLPResNet(28*28, hidden_dim=hidden_dim)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch_idx in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
        test_error, test_loss = epoch(test_dataloader, model, None)

        print(
            f"Epoch {epoch_idx}: Train loss {train_loss:.4f} | Train error {train_error:.4f} | Test loss {test_loss:.4f} | Test error {test_error:.4f}"
        )
    
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
