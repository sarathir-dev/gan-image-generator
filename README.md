# GAN Image Generator

A minimal implementation of a Generative Adversarial Network (GAN) using PyTorch for generating FashionMNIST-style images from random noise. Designed for clarity, modularity, and GPU acceleration.

---

## Features

- Generator and Discriminator architectures implemented with PyTorch
- Adversarial training loop with alternating updates for generator and discriminator
- Evaluation metrics: accuracy, precision, recall, and Fréchet Inception Distance (FID)
- Automatic saving of generated image grids during training for visual inspection
- Easily extensible to other image datasets

---

## Setup Instructions

1. Clone the repository: 
```bash 
git clone https://github.com/sarathir-dev/gan-image-generator.git
```
2. Install required dependencies
3. Run training:
``` train.py ```

---

## Key Components

### Generator

- Transforms random noise vectors into synthetic images
- Uses transposed convolutional layers for upsampling
- Batch normalization and activation functions for stable training

### Discriminator

- Classifies images as real or fake
- Convolutional layers with downsampling
- LeakyReLU activations and dropout for regularization

### Training Loop

- Alternates between training discriminator and generator
- Uses binary cross-entropy loss
- Logs losses and evaluation metrics at each epoch

### Evaluation Metrics

- **Accuracy**: Measures discriminator's ability to distinguish real from fake images
- **Precision & Recall**: Evaluates quality and diversity of generated images
- **FID (Fréchet Inception Distance)**: Quantifies similarity between real and generated image distributions

---

## Results

- Trained model generates FashionMNIST-style images with competitive FID scores
- Sample outputs and training logs are saved in the `results/` directory

---

## Keywords

PyTorch, GAN, Generative Adversarial Network, FashionMNIST, Deep Learning, Image Generation, Computer Vision, Generator, Discriminator, FID, Precision, Recall, Accuracy, Model Evaluation, Python, Machine Learning
