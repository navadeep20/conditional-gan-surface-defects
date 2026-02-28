import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.data_loader_surface import get_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 7

CLASS_NAMES = [
    "normal",
    "scratch",
    "crack",
    "dent",
    "pit",
    "rust",
    "stain",
]

# =============================
# MODEL
# =============================

def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# =============================
# EVALUATION
# =============================

def evaluate():
    loader = get_dataloader(batch_size=32, img_size=64)

    model = get_model()
    model.load_state_dict(torch.load("checkpoints/defect_classifier.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("🔍 Evaluating classifier...")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # =============================
    # METRICS
    # =============================

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    print("\n📉 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

# =============================

if __name__ == "__main__":
    evaluate()