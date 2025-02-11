#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


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
            nn.Linear(hidden_dim, input_dim)
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
    parser = argparse.ArgumentParser(description='Tutorial 04')
    parser.add_argument('--num-samples', '-n', type=int, default=100000, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--hidden-dim', type=int, default=100, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=1, help='Latent dimension')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    args = parser.parse_args()

    # Set arguments
    num_samples = args.num_samples
    seed = args.seed
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    input_dim = 1

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set true parameters
    phi_true = 0.8
    mu1_true = 10.0
    mu2_true = 20.0
    sigma1_true = 3.0
    sigma2_true = 2.0

    # Generate data
    data = []
    for i in range(num_samples):
        if np.random.rand() < phi_true:
            data.append(np.random.normal(mu1_true, sigma1_true))
        else:
            data.append(np.random.normal(mu2_true, sigma2_true))
    data = torch.tensor(data, dtype=torch.float32, requires_grad=False)

    # Dataset
    dataset = data.view(-1, 1)
    dataset = TensorDataset(dataset)
    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define parameters
    model = VAE(input_dim, hidden_dim, latent_dim)

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
    z = torch.randn(num_samples, latent_dim)
    data_posterior = model.p(z).detach().numpy()

    # Plot data and ELBO
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    # Original data
    axs[0].hist(data, bins=50, density=True)
    axs[0].set_title('Data')
    # Generate data
    axs[1].hist(data_posterior, bins=50, density=True)
    axs[1].set_title('Generate data')
    # Training
    axs[2].plot(elbo_list, label='ELBO')
    axs[2].plot(l1_list, label='L1')
    axs[2].plot(l2_list, label='L2')
    axs[2].set_title('Training')
    axs[2].legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
