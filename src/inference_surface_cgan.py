import os
import torch
from torchvision.utils import save_image

from src.generator_cgan_surface import ConditionalGenerator

# =============================
# CONFIG
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOISE_DIM = 100
NUM_CLASSES = 7
IMG_SIZE = 64

CHECKPOINT_PATH = "checkpoints/gen_epoch_4.pth"  # last epoch (since EPOCHS=5)

os.makedirs("samples/inference", exist_ok=True)


# =============================
# LOAD GENERATOR
# =============================

def load_generator():
    gen = ConditionalGenerator(
        noise_dim=NOISE_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    gen.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    gen.eval()
    return gen


# =============================
# GENERATE IMAGES
# =============================

def generate_per_class(num_per_class=8):
    gen = load_generator()

    with torch.no_grad():
        for class_id in range(NUM_CLASSES):
            noise = torch.randn(num_per_class, NOISE_DIM).to(DEVICE)
            labels = torch.full((num_per_class,), class_id, dtype=torch.long).to(DEVICE)

            fake_imgs = gen(noise, labels)

            save_path = f"samples/inference/class_{class_id}.png"

            save_image(
                fake_imgs,
                save_path,
                nrow=4,
                normalize=True
            )

            print(f"✅ Saved {save_path}")


# =============================
# MAIN
# =============================

if __name__ == "__main__":
    generate_per_class()