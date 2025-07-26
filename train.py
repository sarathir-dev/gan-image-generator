import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from models.generator import Generator
from models.discriminator import Discriminator
from utils.plot import plot_generated_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# hyper-parameters
latent_dim = 100
lr = 0.0001
batch_size = 128
epochs = 50

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(1, epochs + 1):
    for i, (imgs, _) in enumerate(dataloader):
        real_imgs = imgs.view(imgs.size(0), -1).to(device)
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

    print(
        f"[Epoch {epoch}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
    if epoch % 10 == 0:
        plot_generated_images(generator, latent_dim, device, epoch)

torch.save(generator.state_dict(), "generator.pth")
