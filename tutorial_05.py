#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms


class QModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(QModel, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.hidden(x)
        mu = self.mu(h)
        log_sigma2 = self.log_sigma2(h)
        return mu, log_sigma2


class PModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(PModel, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        mu = self.mu(z)
        return mu


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.q = QModel(input_dim, hidden_dim, latent_dim)
        self.p = PModel(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        mu_z, log_sigma2_z = self.q(x)
        sigma_z = torch.exp(0.5 * log_sigma2_z)
        z = mu_z + sigma_z * torch.randn_like(sigma_z)
        mu_x = self.p(z)

        l1 = -0.5 * torch.sum((x - mu_x)**2)
        l2 = 0.5 * torch.sum(1.0 + log_sigma2_z - mu_z**2 - sigma_z**2)
        elbo = l1 + l2
        return elbo, l1, l2


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 05')
    parser.add_argument('--num-samples', '-n', type=int, default=64, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden-dim', type=int, default=200, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=20, help='Latent dimension')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default="cpu", help='Device (cpu, cuda, mps)', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    # Set arguments
    num_samples = args.num_samples
    device = args.device
    seed = args.seed
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    input_dim = 784

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Dataset
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)]),
    )
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define parameters
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    elbo_list = []
    l1_list = []
    l2_list = []
    for epoch in range(num_epochs):
        elbo_sum = 0
        l1_sum = 0
        l2_sum = 0
        batch_count = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            elbo, l1, l2 = model(x)
            loss = -elbo
            loss.backward()
            optimizer.step()
            elbo_sum += elbo.item()
            l1_sum += l1.item()
            l2_sum += l2.item()
            batch_count += 1
        elbo_mean = elbo_sum / batch_count
        l1_mean = l1_sum / batch_count
        l2_mean = l2_sum / batch_count
        print(f"Epoch: {epoch+1}/{num_epochs}, ELBO: {elbo_mean:.4f}, L1: {l1_mean:.4f}, L2: {l2_mean:.4f}")
        elbo_list.append(elbo_mean)
        l1_list.append(l1_mean)
        l2_list.append(l2_mean)

    # Generate data from posterior
    z = torch.randn(num_samples, latent_dim).to(device)
    data_posterior = model.p(z).to('cpu')
    images = data_posterior.view(num_samples, 1, 28, 28)
    grid_img = make_grid(
        images,
        nrow=8,
        padding=2,
        normalize=True
    )

    # Plot data and ELBO
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    # Generate data
    axs[0].imshow(grid_img.permute(1, 2, 0))
    axs[0].set_axis_off()
    axs[0].set_title('Generate data')
    # Training
    axs[1].plot(elbo_list, label='ELBO')
    axs[1].plot(l1_list, label='L1')
    axs[1].plot(l2_list, label='L2')
    axs[1].set_title('Training')
    axs[1].legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
