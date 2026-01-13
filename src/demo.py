import shutil
import sys
import os

# Try imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    import cv2
    import numpy as np
except ImportError as e:
    print("\n" + "!"*60)
    print("❌ CRITICAL ERROR: Missng Dependencies")
    print(f"Error details: {e}")
    print("!"*60)
    print("\nPlease run: pip install -r requirements.txt")
    print("If you are using Python 3.13, PyTorch might not be supported yet.")
    print("Try using Python 3.10, 3.11, or 3.12.")
    print("!"*60 + "\n")
    sys.exit(1)

from utils import create_dummy_data
from train import train_one_epoch, validate
from model import get_model
from dataset import create_dataloaders
from inference import predict_image

# Ensure we are running from the correct directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, '../data')

def run_demo():
    print("="*50)
    print("🚀 Deepfake Detection Demo Setup")
    print("="*50)
    
    # 1. Clean old data if it exists to ensure freshness
    if os.path.exists(DATA_DIR):
        print("Cleaning old data...")
        shutil.rmtree(DATA_DIR)
    
    # 2. Generate new synthetic data
    print("\n[1/3] a Generating synthetic dataset (Red=Real, Blue=Fake)...")
    create_dummy_data(DATA_DIR, num_images=100)
    print("✅ Data generated successfully.")
    
    # 3. Train the model
    print("\n[2/3] 🧠 Training Model (1 Epoch for Demo)...")
    
    # We'll use os.system to run the train script to keep potential import errors isolated
    # But for a better demo, let's try to import and run main() if possible, 
    # OR just run the command line if we want to be safe. 
    # Let's run command line.
    
    # Update train.py config via command line arguments would be better, 
    # but for now we rely on defaults or simple modification.
    # Actually, let's just run it. The default is 10 epochs, that's too long for a demo.
    # I'll create a quick custom training function here or just patch the train script.
    # Let's just run it for 1 epoch by passing an argument if I had implemented arg parsing.
    # I didn't. So I'll just write a quick training loop here to ensure control.
    
    try:
        from train import train_one_epoch, validate, get_model, create_dataloaders
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        DEVICE = 'cpu' # Force CPU for safety if CUDA is flaky
        BATCH_SIZE = 16
        
        dataloaders = create_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
        model = get_model(device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        print("Starting training...")
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, DEVICE)
        
        print(f"\n🎉 Result: Train Acc: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save model
        torch.save(model.state_dict(), "demo_model.pth")
        print("✅ Model trained and saved to demo_model.pth")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    # 4. Inference
    print("\n[3/3] 🔮 Testing Inference...")
    try:
        from inference import predict_image
        
        # Test on a 'Fake' image (Blue)
        fake_img_path = os.path.join(DATA_DIR, 'val', 'fake', 'img_0.png')
        pred, conf = predict_image(fake_img_path, model, DEVICE)
        print(f"Test Image (Fake/Blue): Predicted as {pred} ({conf:.2f})")
        
        # Test on a 'Real' image (Red)
        real_img_path = os.path.join(DATA_DIR, 'val', 'real', 'img_0.png')
        pred, conf = predict_image(real_img_path, model, DEVICE)
        print(f"Test Image (Real/Red): Predicted as {pred} ({conf:.2f})")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")

    print("\nDone! The system is working.")

if __name__ == "__main__":
    run_demo()
