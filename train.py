import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torchmetrics.image.fid import FrechetInceptionDistance
from models.generator import Generator
from models.discriminator import Discriminator
from utils.plot import plot_generated_images, plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# hyper-parameters
latent_dim = 100
lr = 0.0001
batch_size = 128
epochs = 10

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

fid = FrechetInceptionDistance(feature=64, normalize=True).to(device)
g_losses, d_losses = [], []
acc_list, prec_list, rec_list, fid_scores = [], [], [], []


for epoch in range(1, epochs + 1):
    all_preds, all_labels = [], []

    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        real_imgs = imgs.view(imgs.size(0), -1)
        valid = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)
        g_loss = criterion(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        with torch.no_grad():
            pred_real = discriminator(real_imgs)
            pred_fake = discriminator(gen_imgs.detach())
            pred_labels = torch.cat(
                [pred_real, pred_fake], dim=0).cpu().numpy()
            true_labels = torch.cat([valid, fake], dim=0).cpu().numpy()

            all_preds.extend((pred_labels > 0.5).astype(int))
            all_labels.extend(true_labels.astype(int))

    # === Metrics ===
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    with torch.no_grad():
        real_imgs_resized = torchvision.transforms.functional.resize(imgs, [
                                                                     299, 299])
        real_imgs_resized = real_imgs_resized.expand(-1, 3, -1, -1)

        gen_imgs_reshaped = gen_imgs.view(-1, 1, 28, 28)
        gen_imgs_resized = torchvision.transforms.functional.resize(
            gen_imgs_reshaped, [299, 299])
        gen_imgs_resized = gen_imgs_resized.expand(-1, 3, -1, -1)

        fid.update(real_imgs_resized, real=True)
        fid.update(gen_imgs_resized, real=False)
        fid_score = fid.compute().item()
        fid.reset()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())
    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)
    fid_scores.append(fid_score)

    print(f"[Epoch {epoch}/{epochs}] "
          f"D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f} | "
          f"Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | FID: {fid_score:.4f}")

    if epoch % 10 == 0 or epoch == epochs:
        print(f"Confusion Matrix:\n{cm}")
        plot_generated_images(generator, latent_dim, device, epoch)

torch.save(generator.state_dict(), "generator.pth")

plot_metrics(g_losses, "Generator Loss Over Epochs",
             "Loss", "g_loss.png")
plot_metrics(d_losses, "Discriminator Loss Over Epochs",
             "Loss", "d_loss.png")
plot_metrics(acc_list, "Discriminator Accuracy Over Epochs",
             "Accuracy", "accuracy.png")
plot_metrics(prec_list, "Precision Over Epochs",
             "Precision", "precision.png")
plot_metrics(rec_list, "Recall Over Epochs", "Recall", "recall.png")
plot_metrics(fid_scores, "FID Score Over Epochs", "FID", "fid.png")
print("Training complete. All metric plots saved plots folder.")
