import os
import torchvision
from PIL import Image

# =============================
# DESTINATION
# =============================

DEST_ROOT = "data/raw"

CLASS_NAMES = [
    "normal",
    "scratch",
    "crack",
    "dent",
    "pit",
    "rust",
    "stain",
]


def main():
    os.makedirs(DEST_ROOT, exist_ok=True)

    # -----------------------------
    # Create class folders
    # -----------------------------
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(DEST_ROOT, cls), exist_ok=True)

    print("Downloading CIFAR10 (one-time)...")

    # -----------------------------
    # Load CIFAR10
    # -----------------------------
    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True
    )

    print("Preparing demo dataset...")

    # -----------------------------
    # Distribute images into classes
    # -----------------------------
    num_images = 2000  # you can increase later

    for i in range(num_images):
        img, _ = dataset[i]  # already PIL image

        class_name = CLASS_NAMES[i % len(CLASS_NAMES)]

        save_path = os.path.join(
            DEST_ROOT,
            class_name,
            f"img_{i}.png"
        )

        img.save(save_path)

    print("✅ Demo dataset ready!")


if __name__ == "__main__":
    main()