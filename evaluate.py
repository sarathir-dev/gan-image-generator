import torch
from models.generator import Generator
from utils.plot import plot_generated_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

latent_dim = 100

generator = Generator(latent_dim).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

plot_generated_images(generator, latent_dim, device, "final")
