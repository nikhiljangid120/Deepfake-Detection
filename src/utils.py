import os
import torch
import numpy as np
import cv2
from pathlib import Path

def create_dummy_data(base_dir, num_images=10):
    """
    Creates dummy data for testing purposes if no data exists.
    Structure:
    base_dir/
      train/
        real/
        fake/
      val/
        real/
        fake/
    """
    splits = ['train', 'val']
    classes = ['real', 'fake']
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(base_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)
            
            # Check if directory is empty
            if not os.listdir(dir_path):
                print(f"Generating {num_images} dummy images in {dir_path}...")
                for i in range(num_images):
                    # Create structured dummy data for easier learning
                    # Real: Reddish images
                    if cls == 'real':
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                        img[:, :, 2] = np.random.randint(200, 255) # High Red channel (BGR in OpenCV)
                        img[:, :, 0:2] = np.random.randint(0, 50)  # Low Blue/Green
                    # Fake: Bluish images
                    else:
                        img = np.zeros((224, 224, 3), dtype=np.uint8)
                        img[:, :, 0] = np.random.randint(200, 255) # High Blue channel
                        img[:, :, 1:] = np.random.randint(0, 50)   # Low Green/Red
                        
                    cv2.imwrite(os.path.join(dir_path, f"img_{i}.png"), img)

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']
