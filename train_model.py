import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import copy
from src.model_utils import get_model

# --- Configuration ---
DATA_DIR = "./data"  # Matches your folder structure
BATCH_SIZE = 32
EPOCHS = 3           # Keep it low for a quick test, increase for better accuracy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"🚀 Starting training on: {DEVICE}")

    # 1. Data Augmentation & Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Data from your ./data folder
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform)
    val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform)
    
    dataloaders = {
        'train': DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    }
    
    print(f"✅ Found {len(train_set)} training images and {len(val_set)} validation images.")
    print(f"✅ Classes: {train_set.classes}")

    # 2. Setup Model
    model = get_model(num_classes=len(train_set.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Accuracy: {epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # 4. Save the Best Model
    print(f"\n🏆 Training Complete. Best Val Acc: {best_acc:.4f}")
    torch.save(best_model_wts, "pneumonia_model.pth")
    print("💾 Model saved as 'pneumonia_model.pth'")

if __name__ == "__main__":
    train()