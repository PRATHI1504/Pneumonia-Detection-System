import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from PIL import Image

# --- 1. The Model Architecture ---
def get_model(num_classes=2):
    """Loads a pre-trained ResNet18 and adjusts it for 2 classes."""
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Change the final layer to output 2 classes (Normal vs Pneumonia)
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 2. Grad-CAM Explainability Tool ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        target_layer.register_backward_hook(self.save_gradient)
        self.target_layer = target_layer

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.eval()
        features = None
        
        # Hook to capture feature maps
        def forward_hook(module, input, output):
            nonlocal features
            features = output
        handle = self.target_layer.register_forward_hook(forward_hook)
        
        output = self.model(x)
        handle.remove()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][torch.argmax(output)] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate Heatmap
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = features.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_grads[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0) # ReLU
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap) # Normalize
            
        return heatmap.numpy()

def overlay_heatmap(img_pil, heatmap):
    """Helper to draw the heatmap on top of the X-Ray"""
    img = np.array(img_pil)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert OpenCV BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend
    superimposed = np.float32(heatmap) * 0.4 + np.float32(img)
    superimposed = superimposed / np.max(superimposed)
    
    return Image.fromarray(np.uint8(255 * superimposed))