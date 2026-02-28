import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from src.data_loader_surface import get_dataloader
from src.generator_cgan_surface import ConditionalGenerator
from src.discriminator_cgan_surface import ConditionalDiscriminator

# =============================
# CONFIG
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NOISE_DIM = 100
NUM_CLASSES = 7
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15   # 🔥 increase a bit for learning
LR = 2e-4

os.makedirs("samples", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# =============================
# INITIALIZE
# =============================

loader = get_dataloader(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

gen = ConditionalGenerator(noise_dim=NOISE_DIM, num_classes=NUM_CLASSES).to(DEVICE)
disc = ConditionalDiscriminator(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(DEVICE)

criterion = nn.BCELoss()

opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))

print(f"Using device: {DEVICE}")

# =============================
# TRAIN LOOP
# =============================

for epoch in range(EPOCHS):
    for batch_idx, (real_imgs, labels) in enumerate(loader):

        real_imgs = real_imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        batch_size = real_imgs.size(0)

        # 🔥 label smoothing
        real_targets = torch.full((batch_size,), 0.9, device=DEVICE)
        fake_targets = torch.full((batch_size,), 0.0, device=DEVICE)

        # ======================
        # Train Discriminator
        # ======================

        noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        fake_imgs = gen(noise, labels)

        # 🔥 small noise to real images (stabilization trick)
        real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)

        disc_real = disc(real_imgs_noisy, labels)
        loss_real = criterion(disc_real, real_targets)

        disc_fake = disc(fake_imgs.detach(), labels)
        loss_fake = criterion(disc_fake, fake_targets)

        loss_disc = (loss_real + loss_fake) / 2

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ======================
        # Train Generator
        # ======================

        output = disc(fake_imgs, labels)
        loss_gen = criterion(output, real_targets)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # ======================
        # Logging
        # ======================

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}/{EPOCHS}] "
                f"Batch [{batch_idx}/{len(loader)}] "
                f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
            )

    # ======================
    # Save sample images
    # ======================

    with torch.no_grad():
        sample_noise = torch.randn(16, NOISE_DIM).to(DEVICE)
        sample_labels = torch.randint(0, NUM_CLASSES, (16,), device=DEVICE)
        fake_samples = gen(sample_noise, sample_labels)

        save_image(
            fake_samples,
            f"samples/epoch_{epoch}.png",
            normalize=True
        )

    # ======================
    # Save checkpoints
    # ======================

    torch.save(gen.state_dict(), f"checkpoints/gen_epoch_{epoch}.pth")
    torch.save(disc.state_dict(), f"checkpoints/disc_epoch_{epoch}.pth")

print("✅ Training finished!")