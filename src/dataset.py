import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Get class folders
        split_dir = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(split_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                        self.samples.append((
                            os.path.join(class_dir, img_name),
                            self.class_to_idx[class_name]
                        ))
        
        print(f"Found {len(self.samples)} images in {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(image_size=224, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
