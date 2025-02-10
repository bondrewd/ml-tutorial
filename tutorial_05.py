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
        self.mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.sigma = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),
        )

    def forward(self, z):
        mu = self.mu(z)
        sigma = self.sigma(z)
        return mu, sigma


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
        mu_z, sigma_z = self.q(x)
        z = mu_z + sigma_z * torch.randn_like(sigma_z)
        mu_x = self.p(z)

        l1 = torch.mean((x - mu_x)**2)
        l2 = torch.mean(sigma_z**2 + mu_z**2 - torch.log(sigma_z**2) - 1)
        loss = l1 + l2
        return loss, l1, l2


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 05')
    parser.add_argument('--num-samples', '-n', type=int, default=100000, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--hidden-dim', type=int, default=100, help='Hidden dimension')
    parser.add_argument('--latent-dim', type=int, default=1, help='Hidden dimension')
    parser.add_argument('--batch-size', type=int, default=1000, help='Hidden dimension')
    args = parser.parse_args()

    # Set arguments
    num_samples = args.num_samples
    seed = args.seed
    lr = args.lr
    num_epochs = args.num_epochs
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    batch_size = args.batch_size

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
    model = VAE(1, hidden_dim, latent_dim)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train model
    loss_list = []
    elbo_list = []
    d_kl_list = []
    for epoch in range(num_epochs):
        loss_sum = 0
        elbo_sum = 0
        d_kl_sum = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            loss, l1, l2 = model(x)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            elbo_sum += l1.item()
            d_kl_sum += l2.item()
        loss_sum /= len(dataloader)
        elbo_sum /= len(dataloader)
        d_kl_sum /= len(dataloader)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum:.2f}, ELBO: {elbo_sum:.2f}, D_KL: {d_kl_sum:.2f}")
        loss_list.append(loss_sum)
        elbo_list.append(elbo_sum)
        d_kl_list.append(d_kl_sum)

    # Generate data from posterior
    z = torch.randn(num_samples, latent_dim)
    mu = model.p(z).detach().numpy()
    data_posterior = np.random.normal(mu, 1)

    # Plot data and ELBO
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    # Original data
    axs[0].hist(data, bins=50, density=True)
    axs[0].set_title('Data')
    # Generate data
    axs[1].hist(data_posterior, bins=50, density=True)
    axs[1].set_title('Generate data')
    # Loss
    axs[2].plot(loss_list, label='Loss')
    axs[2].plot(elbo_list, label='ELBO')

    axs[2].plot(d_kl_list, label='$D_{KL}$')
    axs[2].set_title('Training')
    axs[2].legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
