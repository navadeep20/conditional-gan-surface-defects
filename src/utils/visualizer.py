import matplotlib.pyplot as plt
import torchvision.utils as vutils


def show_batch(images, title="Generated Images"):
    grid = vutils.make_grid(images[:16], nrow=4, normalize=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()