import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# =============================
# LABEL MAP
# =============================

LABEL_MAP = {
    "normal": 0,
    "scratch": 1,
    "crack": 2,
    "dent": 3,
    "pit": 4,
    "rust": 5,
    "stain": 6,
}

# =============================
# DATASET CLASS
# =============================

class SurfaceDefectDataset(Dataset):
    def __init__(self, root_dir="data/raw", img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size

        self.image_paths = []
        self.labels = []

        # transform for GAN (VERY IMPORTANT)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # → [-1, 1]
        ])

        self._load_dataset()

    # -----------------------------
    def _load_dataset(self):
        print("Loading dataset...")

        for class_name in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            if class_name not in LABEL_MAP:
                print(f"⚠️ Skipping unknown class: {class_name}")
                continue

            label = LABEL_MAP[class_name]

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        print(f"✅ Loaded {len(self.image_paths)} images")

    # -----------------------------
    def __len__(self):
        return len(self.image_paths)

    # -----------------------------
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


# =============================
# DATALOADER HELPER
# =============================

def get_dataloader(batch_size=64, img_size=128):
    dataset = SurfaceDefectDataset(img_size=img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # safer for Windows
    )

    return loader


# =============================
# TEST BLOCK
# =============================

if __name__ == "__main__":
    loader = get_dataloader(batch_size=32)

    for imgs, labels in loader:
        print("Batch images shape:", imgs.shape)
        print("Batch labels shape:", labels.shape)
        break