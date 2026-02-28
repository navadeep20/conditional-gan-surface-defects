import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.data_loader_surface import get_dataloader

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

def train_classifier():
    loader = get_dataloader(batch_size=32, img_size=64)

    model = get_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training classifier...")

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

    torch.save(model.state_dict(), "checkpoints/defect_classifier.pth")
    print("✅ Classifier training done!")

# =============================

if __name__ == "__main__":
    train_classifier()