import torch
import torchvision


def show_samples():
    dataset = torchvision.datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)
    data_iter = iter(loader)
    images, _ = next(data_iter)
    torchvision.utils.save_image(
        images, "fashion_samples.png", nrow=5, normalize=True)
    print("Saved sample image grid!")


if __name__ == "__main__":
    show_samples()
