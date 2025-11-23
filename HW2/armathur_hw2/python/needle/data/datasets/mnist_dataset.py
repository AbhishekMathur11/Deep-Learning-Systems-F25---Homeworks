from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.transforms = transforms
        with gzip.open(image_filename, 'rb') as f:
            # Read the magic number and dimensions
            magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
            assert magic == 2051, f"Invalid magic number for images: {magic}"
            
            image_data = f.read()
            self.images = np.frombuffer(image_data, dtype=np.uint8)
            self.images = self.images.reshape(num_images, num_rows, num_cols, 1)
            
            self.images = self.images.astype(np.float32) / 255.0  
        with gzip.open(label_filename, 'rb') as f:

            magic, num_labels = struct.unpack('>II', f.read(8))
            assert magic == 2049, f"Invalid magic number for labels: {magic}"
            
            label_data = f.read()
            self.labels = np.frombuffer(label_data, dtype=np.uint8)
            assert num_images == num_labels, "Number of images and labels do not match"
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        # ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        image = self.images[index]
        label = self.labels[index]

        if self.transforms is not None:
            for transform in self.transforms:
                image = transform(image)
        
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return len(self.images)
        ### END YOUR SOLUTION