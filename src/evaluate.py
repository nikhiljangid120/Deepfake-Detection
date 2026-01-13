import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import create_dataloaders
from model import get_model
from utils import load_checkpoint

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = '../data'

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def main():
    dataloaders = create_dataloaders(DATA_DIR, batch_size=32)
    model = get_model(device=DEVICE)
    
    # Load best model if exists
    try:
        _, _ = load_checkpoint(model, None, path="best_model.pth")
        print("Loaded best model checkpoint.")
    except FileNotFoundError:
        print("Checkpoint not found, using random weights (for testing only).")

    print("Evaluating on Validation Set...")
    labels, preds = evaluate(model, dataloaders['val'], DEVICE)
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['real', 'fake']))
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as confusion_matrix.png")

if __name__ == '__main__':
    main()
