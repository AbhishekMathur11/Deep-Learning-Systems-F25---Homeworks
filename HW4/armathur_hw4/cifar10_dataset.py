import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()

        self.transforms = transforms
        self.p = p

        if train:
            X_batches = []
            y_batches = []
            for i in range(1, 6):
                batch_path = os.path.join(base_folder, f'data_batch_{i}')
                with open(batch_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    X_batches.append(batch[b'data'])
                    y_batches.extend(batch[b'labels'])

            self.X = np.concatenate(X_batches, axis=0).reshape(-1, 3, 32, 32) / 255.0
            self.y = np.array(y_batches)
        else:
            # Load test batch
            test_path = os.path.join(base_folder, 'test_batch')
            with open(test_path, 'rb') as f:
                test_batch = pickle.load(f, encoding='bytes')
                self.X = test_batch[b'data'].reshape(-1, 3, 32, 32) / 255.0
                self.y = np.array(test_batch[b'labels'])
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        """
        img = self.X[index]
        label = self.y[index]
        if self.transforms:
            for transform in self.transforms:
                img = transform(img)
        return img, label
    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
