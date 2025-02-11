#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def log_likelihood(x, phi, p1, p2):
    # Calculate probabilities
    phi = F.sigmoid(phi)
    p1 = F.sigmoid(p1)
    p2 = F.sigmoid(p2)
    # Calculate ELBO
    c1 = phi * (p1*x + (1-p1)*(1-x))
    c2 = (1-phi) * (p2*x + (1-p2)*(1-x))
    return torch.sum(torch.log(c1 + c2))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 01')
    parser.add_argument('--num-samples', '-n', type=int, default=100000, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=200, help='Number of epochs')
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
    p1_true = 0.6
    p2_true = 0.3
    # Generate data
    data = []
    for i in range(num_samples):
        if np.random.rand() < phi_true:
            if np.random.rand() < p1_true:
                data.append(1)
            else:
                data.append(0)
        else:
            if np.random.rand() < p2_true:
                data.append(1)
            else:
                data.append(0)
    data = torch.tensor(data, dtype=torch.float32, requires_grad=False)

    # Define parameters
    phi = nn.Parameter(torch.randn(1), requires_grad=True)
    p1 = nn.Parameter(torch.randn(1), requires_grad=True)
    p2 = nn.Parameter(torch.randn(1), requires_grad=True)

    # Define optimizer
    optimizer = torch.optim.Adam([phi, p1, p2], lr=lr)

    # Train model
    loss_list = []
    phi_list = [F.sigmoid(phi).data.item()]
    p1_list = [F.sigmoid(p1).data.item()]
    p2_list = [F.sigmoid(p2).data.item()]
    for _ in range(num_epochs):
        # Train
        optimizer.zero_grad()
        loss = -log_likelihood(data, phi, p1, p2) / num_samples
        loss.backward()
        optimizer.step()
        # Save ELBO and parameters
        loss_list.append(loss.item())
        phi_list.append(torch.sigmoid(phi).data.item())
        p1_list.append(torch.sigmoid(p1).data.item())
        p2_list.append(torch.sigmoid(p2).data.item())

    # Print results
    print(f"phi: {torch.sigmoid(phi).data.item()}")
    print(f"p1:  {torch.sigmoid(p1).data.item()}")
    print(f"p2:  {torch.sigmoid(p2).data.item()}")

    # Generate data from posterior
    data_posterior = []
    for i in range(num_samples):
        if np.random.rand() < F.sigmoid(phi).data.item():
            if np.random.rand() < F.sigmoid(p1).data.item():
                data_posterior.append(1)
            else:
                data_posterior.append(0)
        else:
            if np.random.rand() < F.sigmoid(p2).data.item():
                data_posterior.append(1)
            else:
                data_posterior.append(0)

    # Plot data and ELBO
    fig, axs = plt.subplots(2,2)
    # Original data
    axs[0,0].hist(data, bins=2)
    axs[0,0].set_title('Data')
    # Generate data
    axs[1,0].hist(data_posterior, bins=2)
    axs[1,0].set_title('Generate data')
    # Loss
    axs[0,1].plot(loss_list)
    axs[0,1].set_title('Loss')
    # Parameters
    axs[1,1].plot(phi_list, label='phi')
    axs[1,1].plot(p1_list, label='p1')
    axs[1,1].plot(p2_list, label='p2')
    axs[1,1].legend()
    axs[1,1].set_title('Parameters')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
