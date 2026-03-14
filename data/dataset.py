import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

# Try to import albumentations for advanced augmentation
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ASL_LABELS, ASL_LABELS_INV, DataConfig


class ASLDataset(Dataset):

    
    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):

        self.manifest_path = manifest_path
        self.transform = transform
        self.target_size = target_size
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            self.samples = json.load(f)
        
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        sample = self.samples[idx]
        image_path = sample["path"]
        label = sample["label"]
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize and normalize
            image = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])(image)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
 
        class_counts = {}
        for sample in self.samples:
            label = sample["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total = sum(class_counts.values())
        num_classes = max(class_counts.keys()) + 1
        
        weights = torch.zeros(num_classes)
        for label, count in class_counts.items():
            weights[label] = total / (num_classes * count)
        
        return weights
    
    def get_sample_weights(self) -> List[float]:
    
        class_weights = self.get_class_weights()
        return [class_weights[s["label"]].item() for s in self.samples]


class SignMNISTDataset(Dataset):
    
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.csv_path = csv_path
        self.transform = transform
        self.target_size = target_size
        
        # Load CSV
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        self.labels = df.iloc[:, 0].values
        self.pixels = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        
        print(f"Loaded {len(self.labels)} samples from {csv_path}")
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = int(self.labels[idx])
        pixels = self.pixels[idx]
        
        # Convert to PIL Image (grayscale to RGB)
        image = Image.fromarray(pixels, mode='L').convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])(image)
        
        return image, label


def get_train_transforms(image_size: int = 224) -> transforms.Compose:

    return transforms.Compose([
        # Spatial transforms
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10,
        ),
        
        # Color transforms
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize (ImageNet statistics)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        # Random erasing (CutOut)
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value='random',
        ),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Only resize and normalize - we want consistent evaluation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_albumentations_transforms(image_size: int = 224, is_train: bool = True):
    """
    Advanced augmentation using albumentations library.
    
    Albumentations is faster than torchvision and has more augmentations.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Install albumentations: pip install albumentations")
    
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5
            ),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.GaussianBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 8,
                max_width=image_size // 8,
                p=0.25
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    test_manifest: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 16,
    image_size: int = 224,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
) -> Dict[str, DataLoader]:


    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    train_dataset = ASLDataset(
        manifest_path=train_manifest,
        transform=train_transform,
        target_size=(image_size, image_size),
    )
    
    val_dataset = ASLDataset(
        manifest_path=val_manifest,
        transform=val_transform,
        target_size=(image_size, image_size),
    )
    
    train_sampler = None
    shuffle_train = True
    
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle_train = False  
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
    }
    
    if test_manifest:
        test_dataset = ASLDataset(
            manifest_path=test_manifest,
            transform=val_transform,
            target_size=(image_size, image_size),
        )
        
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    return loaders


def show_batch(dataloader: DataLoader, num_images: int = 16):

    import matplotlib.pyplot as plt
    
    # Get one batch
    images, labels = next(iter(dataloader))
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i >= num_images or i >= len(images):
            ax.axis('off')
            continue
        
        img = images[i].permute(1, 2, 0).numpy()
        label = ASL_LABELS_INV[labels[i].item()]
        
        ax.imshow(img)
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('batch_visualization.png')
    print("Saved batch visualization to batch_visualization.png")


if __name__ == "__main__":
    print("Testing ASL Dataset...")
    
    test_manifest = [
        {"path": "test.jpg", "label": 0},
        {"path": "test.jpg", "label": 1},
    ]
    
    os.makedirs("./data/processed", exist_ok=True)
    with open("./data/processed/test_manifest.json", "w") as f:
        json.dump(test_manifest, f)
    
    print("\nTrain transforms:")
    train_tf = get_train_transforms()
    print(train_tf)
    
    print("\nVal transforms:")
    val_tf = get_val_transforms()
    print(val_tf)
    
    print("\nDataset module loaded successfully!")
