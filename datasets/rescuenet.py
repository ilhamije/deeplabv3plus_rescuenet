import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class RescueNetDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Define image and label paths
        self.images = []
        self.labels = []

        image_dir = os.path.join(
            root, 'train' if split == 'train' else 'val', 'train-org-img')
        label_dir = os.path.join(
            root, 'train' if split == 'train' else 'val', 'train-label-img')

        # Collect image and label file paths
        for img_file in os.listdir(image_dir):
            if img_file.endswith('.jpg'):
                self.images.append(os.path.join(image_dir, img_file))
                label_file = img_file.replace('.jpg', '_lab.png')
                self.labels.append(os.path.join(label_dir, label_file))

        # Define color mapping for labels
        self.color_map = {
            0: (0, 0, 0),         # Unlabeled
            1: (61, 230, 250),    # Water
            2: (180, 120, 120),   # Building - No Damage
            3: (235, 255, 7),     # Building - Medium Damage
            4: (255, 184, 6),     # Building - Major Damage
            5: (255, 0, 0),       # Building - Total Destruction
            6: (255, 0, 245),     # Vehicle
            7: (140, 140, 140),   # Road - Clear
            8: (160, 150, 20),    # Road - Blocked
            9: (4, 250, 7),       # Tree
            10: (255, 235, 0),    # Pool
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        # Convert label image to color mapping
        label = self.colorize_label(label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

    def colorize_label(self, label):
        """Convert label indices to color mapping."""
        # Create an empty image with RGB mode
        colorized_label = Image.new("RGB", label.size)

        # Map each pixel to the corresponding color
        for label_index, color in self.color_map.items():
            # Create a mask for the current label index
            mask = np.array(label) == label_index
            colorized_label_np = np.array(colorized_label)
            colorized_label_np[mask] = color

        return Image.fromarray(colorized_label_np)

    @classmethod
    def decode_target(cls, mask):
        """Decode a segmentation mask into an RGB image."""
        return cls.cmap[mask]


def load_rescuenet_dataset(root, batch_size=4, split='train', transforms=None):
    """
    Create a DataLoader for the RescueNet dataset.

    Args:
        root (str): Root directory of the dataset.
        batch_size (int): Number of samples per batch.
        split (str): Split of the dataset, e.g., 'train', 'val'.
        transforms (callable, optional): Transformations to apply to the dataset.

    Returns:
        DataLoader: A DataLoader instance for the dataset.
    """
    # Initialize the RescueNetDataset with the correct split argument
    dataset = RescueNetDataset(root=root, split=split, transform=transforms)

    # Create a DataLoader for the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)  # num_workers for parallel data loading

    return dataloader
