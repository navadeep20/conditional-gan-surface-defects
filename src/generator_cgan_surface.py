import torch
import torch.nn as nn

# =============================
# GENERATOR (CPU OPTIMIZED)
# =============================

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=7, img_channels=3, feature_g=64, embed_dim=50):
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        input_dim = noise_dim + embed_dim

        # Project and reshape (for 64x64 output)
        self.net = nn.Sequential(
            nn.Linear(input_dim, feature_g * 8 * 4 * 4),
            nn.BatchNorm1d(feature_g * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Upsampling blocks → output 64x64
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels)
        x = torch.cat([noise, label_embed], dim=1)

        x = self.net(x)
        x = x.view(x.size(0), 512, 4, 4)

        img = self.conv_blocks(x)
        return img


# =============================
# TEST
# =============================

if __name__ == "__main__":
    noise = torch.randn(16, 100)
    labels = torch.randint(0, 7, (16,))

    gen = ConditionalGenerator()
    out = gen(noise, labels)
    print("Generated image shape:", out.shape)