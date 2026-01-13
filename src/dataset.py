import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size=224):
    """
    Returns training and validation transforms.
    
    Args:
        img_size (int): Input image size for the model (default: 224 for ViT).
    """
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms

def create_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Creates DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir (str): Root directory containing 'train', 'val', (and optionally 'test') subdirectories.
        batch_size (int): Batch size.
        num_workers (int): Number of worker threads.
        
    Returns:
        dict: Dictionary containing dataloaders for available splits.
    """
    dataloaders = {}
    train_transforms, val_transforms = get_transforms()

    # Train Split
    train_path = os.path.join(data_dir, 'train')
    if os.path.exists(train_path):
        train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
        dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        print(f"Found {len(train_dataset)} training images. Classes: {train_dataset.classes}")
    else:
        print(f"Warning: Train directory not found at {train_path}")

    # Validation Split
    val_path = os.path.join(data_dir, 'val')
    if os.path.exists(val_path):
        val_dataset = datasets.ImageFolder(val_path, transform=val_transforms)
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Found {len(val_dataset)} validation images. Classes: {val_dataset.classes}")

    # Test Split (Optional)
    test_path = os.path.join(data_dir, 'test')
    if os.path.exists(test_path):
        test_dataset = datasets.ImageFolder(test_path, transform=val_transforms)
        dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Found {len(test_dataset)} test images.")

    return dataloaders
