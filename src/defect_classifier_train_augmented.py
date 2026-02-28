import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import ConcatDataset

from src.data_loader_surface import SurfaceDefectDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 7
EPOCHS = 5
LR = 1e-3

os.makedirs("checkpoints", exist_ok=True)

# =============================
# MODEL
# =============================

def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# =============================
# TRAIN
# =============================

def train_augmented():
    print("📦 Loading real dataset...")
    real_dataset = SurfaceDefectDataset(root_dir="data/raw", img_size=64)

    print("📦 Loading synthetic dataset...")
    synthetic_dataset = SurfaceDefectDataset(root_dir="data/synthetic", img_size=64)

    combined_dataset = ConcatDataset([real_dataset, synthetic_dataset])

    loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    model = get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training augmented classifier...")

    for epoch in range(EPOCHS):
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/defect_classifier_augmented.pth")
    print("✅ Augmented classifier training done!")

# =============================

if __name__ == "__main__":
    train_augmented()