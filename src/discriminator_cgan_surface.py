import torch
import torch.nn as nn

# =============================
# DISCRIMINATOR (CPU OPTIMIZED)
# =============================

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=7, img_channels=3, feature_d=64, img_size=64):
        super().__init__()

        self.img_size = img_size

        # Label embedding → spatial map
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_d, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, 1, 4, 1, 0),  # 1x1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        batch_size = img.size(0)

        label_embed = self.label_emb(labels)
        label_embed = label_embed.view(batch_size, 1, self.img_size, self.img_size)

        x = torch.cat([img, label_embed], dim=1)
        validity = self.net(x)

        validity = validity.view(validity.size(0), -1)
        validity = validity.mean(dim=1)

        return validity


# =============================
# TEST
# =============================

if __name__ == "__main__":
    imgs = torch.randn(16, 3, 64, 64)
    labels = torch.randint(0, 7, (16,))

    disc = ConditionalDiscriminator()
    out = disc(imgs, labels)
    print("Discriminator output shape:", out.shape)