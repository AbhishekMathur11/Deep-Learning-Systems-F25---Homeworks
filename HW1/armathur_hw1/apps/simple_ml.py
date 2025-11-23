"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(label_filename, 'rb') as f:
      _, num_labels = struct.unpack(">II", f.read(8))
      label_data = np.frombuffer(f.read(), dtype=np.uint8)
    with gzip.open(image_filesname, 'rb') as f:
      _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
      image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    
    parsed_array = []
    for img in image_data:
      arr = np.array(img).astype(np.float32)
      arr = arr.flatten()
      arr_norm = arr/255
      parsed_array.append(arr_norm)
    label_array = []
    for label in label_data:
      label_array.append(label)
    X = np.array(parsed_array, dtype=np.float32)
    y = np.array(label_array, dtype=np.uint8)
    return X,y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    m = Z.shape[0]
    log_term = ndl.log(ndl.summation(ndl.exp(Z), axes = (1,)))
    h_y = ndl.summation(Z*y_one_hot, axes = (1,))

    l_sm = log_term - h_y
    return ndl.summation(l_sm, axes = (0,)) * (1.0 / m)
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    n = X.shape[0]
    k = W2.shape[1]

    for i in range(0, n, batch):
       Xb = ndl.Tensor(X[i: i + batch], requires_grad = False)
       yb = y[i: i + batch]
       B = Xb.shape[0]

       Z1 = ndl.matmul(Xb, W1)
       H = ndl.relu(Z1)
       Z2 = ndl.matmul(H, W2)

    
       I_y = np.zeros((B,k), dtype=np.float32)
       I_y[np.arange(B), yb] = 1

       Yb = ndl.Tensor(I_y, requires_grad=False)

       loss = softmax_loss(Z2, Yb)

       loss.backward()

       W1.data -= lr*(W1.grad.numpy())
       W2.data -= lr*(W2.grad.numpy())

       W1.grad = None
       W2.grad = None

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
