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
        self.root_dir = root_dir  # Keep the root_dir as is
        self.split = split  # Store the split value (train or val)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Print the path being used for debugging
        print(f"Using root directory: {self.root_dir}")

        split_dir = os.path.join(self.root_dir, self.split)  # Use split dir correctly
        print(f"Using split directory: {split_dir}")  # Debugging line

        # Add error checking
        if not os.path.exists(split_dir):
            raise ValueError(f"Directory not found: {split_dir}")
            
        class_dirs = sorted(os.listdir(split_dir))
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {split_dir}")
            
        print(f"Found {len(class_dirs)} classes in {split_dir}")  # Debug info
        
        for class_id, class_name in enumerate(class_dirs):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
                if len(image_files) == 0:
                    print(f"Warning: No images found in {class_dir}")
                    continue
                    
                for image_name in image_files:
                    self.image_paths.append(os.path.join(class_dir, image_name))
                    self.labels.append(class_id)
                    
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {split_dir}")


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
            transforms.RandAugment(num_ops=1, magnitude=5),  # Reduced augmentation intensity
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
