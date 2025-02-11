#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def elbo(x, phi, mu1, mu2, sigma1, sigma2):
    # Calculate probabilities
    phi = F.sigmoid(phi)
    sigma1 = F.softplus(sigma1)
    sigma2 = F.softplus(sigma2)
    # Calculate ELBO
    c1 = torch.exp(-0.5*(x-mu1)**2/sigma1**2) / (sigma1 * np.sqrt(2*np.pi)) * phi
    c2 = torch.exp(-0.5*(x-mu2)**2/sigma2**2) / (sigma2 * np.sqrt(2*np.pi)) * (1-phi)
    return torch.mean(torch.log(c1 + c2 + 1e-10))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 02')
    parser.add_argument('--num-samples', '-n', type=int, default=100000, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1000, help='Number of epochs')
    args = parser.parse_args()

    # Set arguments
    num_samples = args.num_samples
    seed = args.seed
    lr = args.lr
    num_epochs = args.num_epochs

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

    # Define parameters
    # Bad guess: 300000, lr=0.0001
    # phi = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    # mu1 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    # mu2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    # sigma1 = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    # sigma2 = nn.Parameter(torch.tensor(2.0), requires_grad=True)

    # So-so guess: 3000 epochs, lr=0.01
    # phi = nn.Parameter(torch.tensor(0.5), requires_grad=True)
    # mu1 = nn.Parameter(torch.tensor(15.0), requires_grad=True)
    # mu2 = nn.Parameter(torch.tensor(15.0), requires_grad=True)
    # sigma1 = nn.Parameter(torch.tensor(4.0), requires_grad=True)
    # sigma2 = nn.Parameter(torch.tensor(6.0), requires_grad=True)

    # Good guess: 1000 epochs, lr=0.01
    phi = nn.Parameter(torch.tensor(0.7), requires_grad=True)
    mu1 = nn.Parameter(torch.tensor(13.0), requires_grad=True)
    mu2 = nn.Parameter(torch.tensor(26.0), requires_grad=True)
    sigma1 = nn.Parameter(torch.tensor(5.0), requires_grad=True)
    sigma2 = nn.Parameter(torch.tensor(2.0), requires_grad=True)

    # Define optimizer
    optimizer = torch.optim.Adam([phi, mu1, mu2, sigma1, sigma2], lr=lr)

    # Train model
    loss_list = []
    phi_list = [F.sigmoid(phi).data.item()]
    mu1_list = [mu1.data.item()]
    mu2_list = [mu2.data.item()]
    sigma1_list = [sigma1.data.item()]
    sigma2_list = [sigma2.data.item()]
    for _ in range(num_epochs):
        # Train
        optimizer.zero_grad()
        loss = -elbo(data, phi, mu1, mu2, sigma1, sigma2)
        loss.backward()
        optimizer.step()
        # Save ELBO and parameters
        loss_list.append(loss.item())
        phi_list.append(torch.sigmoid(phi).data.item())
        mu1_list.append(mu1.data.item())
        mu2_list.append(mu2.data.item())
        sigma1_list.append(sigma1.data.item())
        sigma2_list.append(sigma2.data.item())

    # Print results
    print(f"phi: {F.sigmoid(phi).data.item():.2f}")
    print(f"mu1: {mu1.data.item():.2f}")
    print(f"mu2: {mu2.data.item():.2f}")
    print(f"sigma1: {sigma1.data.item():.2f}")
    print(f"sigma2: {sigma2.data.item():.2f}")

    # Generate data from posterior
    data_posterior = []
    for i in range(num_samples):
        if np.random.rand() < F.sigmoid(phi).data.item():
            data_posterior.append(np.random.normal(mu1.data.item(), sigma1.data.item()))
        else:
            data_posterior.append(np.random.normal(mu2.data.item(), sigma2.data.item()))

    # Plot data and ELBO
    fig, axs = plt.subplots(2,2, figsize=(8,8))
    # Original data
    axs[0,0].hist(data, bins=50, density=True)
    axs[0,0].set_title('Data')
    # Generate data
    axs[1,0].hist(data_posterior, bins=50, density=True)
    axs[1,0].set_title('Generate data')
    # Loss
    axs[0,1].plot(loss_list)
    axs[0,1].set_title('Loss')
    # Parameters
    axs[1,1].plot(mu1_list, label='mu1')
    axs[1,1].plot(mu2_list, label='mu2')
    axs[1,1].plot(sigma1_list, label='sigma1')
    axs[1,1].plot(sigma2_list, label='sigma2')
    axs[1,1].legend()
    axs[1,1].set_title('Parameters')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
