import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ImageNetLocalDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the root directory of the ImageNet dataset.
            split (str): "train" or "val" to specify which dataset to use.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_id, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path)

        # Convert grayscale to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config, is_train=True):
    """
    Args:
        config: Configuration object containing dataset and training parameters.
        is_train (bool): Whether to apply training or validation transformations.

    Returns:
        A torchvision.transforms.Compose object with the desired transformations.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
