import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image


class RescueNetDataset(data.Dataset):
    # Color mapping from the original paper
    cmap = np.array([
        [0, 0, 0],        # Unlabeled
        [61, 230, 250],   # Water
        [180, 120, 120],  # Building - No Damage
        [235, 255, 7],    # Building - Medium Damage
        [255, 184, 6],    # Building - Major Damage
        [255, 0, 0],      # Building - Total Destruction
        [255, 0, 245],    # Vehicle
        [140, 140, 140],  # Road - Clear
        [160, 150, 20],   # Road - Blocked
        [4, 250, 7],      # Tree
        [255, 235, 0],    # Pool
    ], dtype=np.uint8)

    def __init__(self, root, image_set='train', transforms=None):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms

        # Paths to directories following VOC structure
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.label_dir = os.path.join(self.root, 'SegmentationClass')
        self.split_file = os.path.join(
            self.root, 'ImageSets', 'Segmentation', f'{self.image_set}.txt')

        # Load image IDs from the split file
        with open(self.split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        # Construct full paths for the image and the corresponding label mask
        img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        lbl_path = os.path.join(self.label_dir, f'{image_id}_lab.png')

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)

        # Apply any provided transformations (if available)
        if self.transforms:
            img, lbl = self.transforms(img, lbl)

        # Convert label to a numpy array for easier processing
        lbl = np.array(lbl, dtype=np.int64)

        return img, lbl

    @classmethod
    def decode_target(cls, mask):
        """Decode a segmentation mask into an RGB image."""
        return cls.cmap[mask]


def load_rescuenet_dataset(root, batch_size=4, image_set='train', transforms=None):
    """
    Create a DataLoader for the RescueNet dataset.

    Args:
        root (str): Root directory of the dataset.
        batch_size (int): Number of samples per batch.
        image_set (str): Split of the dataset, e.g., 'train', 'val'.
        transforms (callable, optional): Transformations to apply to the dataset.

    Returns:
        DataLoader: A DataLoader instance for the dataset.
    """
    dataset = RescueNetDataset(
        root=root, image_set=image_set, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    return dataloader
