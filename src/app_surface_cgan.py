import sys
import os

# 🔥 Fix for Streamlit module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import io
import torch
import streamlit as st
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from PIL import Image
import zipfile

from src.generator_cgan_surface import ConditionalGenerator

# =============================
# CONFIG
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_DIM = 100
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

CHECKPOINT_PATH = "checkpoints/gen_epoch_14.pth"

# =============================
# LOAD GENERATOR (cached)
# =============================

@st.cache_resource
def load_generator():
    gen = ConditionalGenerator(
        noise_dim=NOISE_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    gen.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    gen.eval()
    return gen

gen = load_generator()

# =============================
# STREAMLIT UI
# =============================

st.title("🛠️ Conditional GAN — Surface Defect Generator")
st.write("Generate synthetic manufacturing defects on demand.")

defect_type = st.selectbox("Select Defect Type", CLASS_NAMES)
num_images = st.slider("Number of Images", 1, 64, 16)

if st.button("🚀 Generate Samples"):

    class_id = CLASS_NAMES.index(defect_type)

    noise = torch.randn(num_images, NOISE_DIM).to(DEVICE)
    labels = torch.full((num_images,), class_id, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        fake_imgs = gen(noise, labels).cpu()

    # Show grid
    grid = make_grid(fake_imgs, nrow=4, normalize=True)
    st.image(grid.permute(1, 2, 0).numpy(), caption="Generated Samples")

    # =============================
    # ZIP DOWNLOAD
    # =============================

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, img_tensor in enumerate(fake_imgs):
            img = TF.to_pil_image((img_tensor + 1) / 2)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            zf.writestr(f"{defect_type}_{i}.png", img_bytes.getvalue())

    st.download_button(
        label="📥 Download Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="synthetic_defects.zip",
        mime="application/zip",
    )