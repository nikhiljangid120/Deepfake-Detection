import torch
import torch.nn as nn
from torchvision import models

class DeepfakeViT(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        Vision Transformer (ViT) model for Deepfake Detection.
        
        Args:
            num_classes (int): Number of output classes (2 for Real vs Fake).
            pretrained (bool): Whether to use pretrained weights.
        """
        super(DeepfakeViT, self).__init__()
        
        # Load Pretrained ViT-B/16
        # weights='IMAGENET1K_V1' if pretrained else None
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.vit_b_16(weights=weights)
        
        # Modify the classification head
        # The original head is stored in self.model.heads.head
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def get_model(device='cpu'):
    model = DeepfakeViT(num_classes=2, pretrained=True)
    model = model.to(device)
    return model
