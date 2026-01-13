import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import create_dataloaders
from model import get_model
from utils import save_checkpoint, create_dummy_data

# Configurations
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
DATA_DIR = '../data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def main():
    print(f"Using device: {DEVICE}")
    
    # Check for data, if not exist, create dummy data
    if not os.path.exists(os.path.join(DATA_DIR, 'train')):
        print("Data directory not found. Creating dummy data for testing...")
        create_dummy_data(DATA_DIR, num_images=50)

    dataloaders = create_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    model = get_model(device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, path="best_model.pth")
            
    print("Training Complete.")

if __name__ == '__main__':
    main()
