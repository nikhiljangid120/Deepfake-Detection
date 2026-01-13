import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from utils import load_checkpoint
import argparse

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    class_names = ['fake', 'real'] # ImageFolder sorts alphabetically: fake=0, real=1
    prediction = class_names[predicted.item()]
    
    return prediction, confidence.item()

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(device=device)
    
    try:
        load_checkpoint(model, None, path="best_model.pth")
    except FileNotFoundError:
        print("Warning: No checkpoint found, using random weights.")
        
    pred, conf = predict_image(args.image, model, device)
    print(f"Prediction: {pred} (Confidence: {conf:.4f})")

if __name__ == '__main__':
    main()
