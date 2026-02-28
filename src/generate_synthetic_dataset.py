import os
import torch
from torchvision.utils import save_image

from src.generator_cgan_surface import ConditionalGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOISE_DIM = 100
NUM_CLASSES = 7
IMAGES_PER_CLASS = 200  # small but enough

SAVE_ROOT = "data/synthetic"

CLASS_NAMES = [
    "normal",
    "scratch",
    "crack",
    "dent",
    "pit",
    "rust",
    "stain",
]

os.makedirs(SAVE_ROOT, exist_ok=True)


def load_generator():
    gen = ConditionalGenerator(noise_dim=NOISE_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    gen.load_state_dict(torch.load("checkpoints/gen_epoch_14.pth", map_location=DEVICE))
    gen.eval()
    return gen


def generate_dataset():
    gen = load_generator()

    print("🚀 Generating synthetic dataset...")

    with torch.no_grad():
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(SAVE_ROOT, class_name)
            os.makedirs(class_dir, exist_ok=True)

            for i in range(IMAGES_PER_CLASS):
                noise = torch.randn(1, NOISE_DIM).to(DEVICE)
                label = torch.tensor([class_id], device=DEVICE)

                fake_img = gen(noise, label)

                save_path = os.path.join(class_dir, f"fake_{i}.png")
                save_image(fake_img, save_path, normalize=True)

            print(f"✅ Generated class: {class_name}")

    print("✅ Synthetic dataset ready!")


if __name__ == "__main__":
    generate_dataset()