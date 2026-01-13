import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
from utils import load_checkpoint

# 1. Setup Model
DEVICE = 'cpu' # Use CPU for safety in demo
model = get_model(device=DEVICE)

# Try to load the demo model first, then best_model
try:
    if torch.cuda.is_available(): # Just in case
        checkpoint = torch.load("demo_model.pth", map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load("demo_model.pth", map_location=torch.device('cpu'))
    
    # Check if checkpoint is a state_dict or full checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # It might be just the state dict
    print("✅ Loaded demo_model.pth")
except:
    try:
        load_checkpoint(model, None, path="best_model.pth")
        print("✅ Loaded best_model.pth")
    except:
        print("⚠️ No model found. Using random weights (Predictions will be random).")

model.eval()

# 2. Define Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Prediction Function
def predict(image):
    if image is None:
        return "Please upload an image."
    
    image = Image.fromarray(image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # Classes: 0=fake, 1=real (Alphabetical)
    # Return dictionary for Gradio Label
    return {
        "Fake": float(probabilities[0][0]),
        "Real": float(probabilities[0][1])
    }

# 4. Create Gradio UI
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown("# 🕵️ Deepfake Image Detection System")
    gr.Markdown("Upload an image to check if it's **Real** or **Fake**.")

    with gr.Row():
        gr.Markdown(
            """
            > ⚠️ **DEMO MODE WARNING**
            > This model is currently trained on **Synthetic Data** (Red Colors = Real, Blue Colors = Fake) because no real dataset was provided.
            > **It does not detect actual faces yet.** usage on real photos will be random or based on the photo's color.
            """
        )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Input Image")
            predict_btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            # Label component shows top classes and confidence
            label_output = gr.Label(num_top_classes=2, label="Prediction Result")
    
    predict_btn.click(predict, inputs=input_img, outputs=label_output)
    
    gr.Markdown("---")
    gr.Markdown("### How it works")
    gr.Markdown("This system uses a **Vision Transformer (ViT)** to analyze patterns in the image. "
                "It has been trained to distinguish between real photos and AI-generated deepfakes.")

# Launch
if __name__ == "__main__":
    demo.launch(inbrowser=True)
