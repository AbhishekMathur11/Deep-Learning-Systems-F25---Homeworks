import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.batch_idx = 0

        if self.shuffle:
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, range(self.batch_size, len(self.dataset), self.batch_size))


        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.batch_idx >= len(self.ordering):
            raise StopIteration

        batch = self.ordering[self.batch_idx]
        self.batch_idx += 1

        batch_data = []
        for idx in batch:
            batch_data.append(self.dataset[idx])

        if len(batch_data) > 0:

            if isinstance(batch_data[0], tuple) and len(batch_data[0]) >= 2:
                
                batch_inputs = [item[0] for item in batch_data]
                batch_labels = [item[1] for item in batch_data]
                
                batch_inputs = Tensor(np.stack(batch_inputs))
                batch_labels = Tensor(np.stack(batch_labels))

                return batch_inputs, batch_labels
            else:
                if isinstance(batch_data[0], tuple):
                    batch_inputs = Tensor(np.stack([item[0] for item in batch_data]))
                else:
                    batch_inputs = Tensor(np.stack(batch_data))
                return (batch_inputs,)
        else:
            raise StopIteration
        ### END YOUR SOLUTION

