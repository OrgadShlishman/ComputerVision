"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> (torch.tensor, int):
        """Get a sample and label from the dataset."""
        if index < len(self.real_image_names):
            label = 0
            label_str = 'real'
        else:
            label = 1
            label_str = 'fake'
        image_names_list = self.real_image_names + self.fake_image_names
        image_name = os.path.join(self.root_path, label_str, image_names_list[index])
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        image = image.detach().clone()
        return (image, label)

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)
