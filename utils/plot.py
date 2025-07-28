import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import os


def plot_generated_images(generator, latent_dim, device, epoch):
    z = torch.randn(25, latent_dim).to(device)
    gen_imgs = generator(z).view(-1, 1, 28, 28).cpu().detach()
    grid = torchvision.utils.make_grid(gen_imgs, nrow=5, normalize=True)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.title(f'Generated Images @ Epoch {epoch}')
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig(f"gen_epoch_{epoch}.png")
    plt.close()


def plot_metrics(metric_list, title, ylabel, filename):
    os.makedirs('plots', exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(metric_list) + 1), metric_list, marker='o')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f'plots/{filename}')
    plt.close()
