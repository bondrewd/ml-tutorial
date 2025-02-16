#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def likelihood(x, gamma, thetaA, thetaB):
    p_x_z1 = gamma * thetaA ** x * (1 - thetaA) ** (1 - x)
    p_x_z0 = (1 - gamma) * thetaB ** x * (1 - thetaB) ** (1 - x)
    p_x = p_x_z1 + p_x_z0
    likelihood = torch.prod(p_x)
    return likelihood


def log_likelihood(x, gamma, thetaA, thetaB):
    p_x_z1 = gamma * thetaA ** x * (1 - thetaA) ** (1 - x)
    p_x_z0 = (1 - gamma) * thetaB ** x * (1 - thetaB) ** (1 - x)
    p_x = p_x_z1 + p_x_z0
    log_p_x = torch.log(p_x)
    likelihood = torch.sum(log_p_x)
    return likelihood


def log_likelihood_with_elbo_and_bad_q(x, gamma, thetaA, thetaB):
    p_x_z1 = gamma * thetaA ** x * (1 - thetaA) ** (1 - x)
    p_x_z0 = (1 - gamma) * thetaB ** x * (1 - thetaB) ** (1 - x)
    q_z1 = 0.5
    q_z0 = 0.5
    elbo_x = q_z1 * torch.log(p_x_z1 / q_z1) + q_z0 * torch.log(p_x_z0 / q_z0)
    elbo = torch.sum(elbo_x)
    return elbo


def log_likelihood_with_elbo_and_good_q(x, gamma, thetaA, thetaB):
    p_x_z1 = gamma * thetaA ** x * (1 - thetaA) ** (1 - x)
    p_x_z0 = (1 - gamma) * thetaB ** x * (1 - thetaB) ** (1 - x)
    p_x = p_x_z1 + p_x_z0
    q_z1 = p_x_z1 / p_x
    q_z0 = p_x_z0 / p_x
    elbo_x = q_z1 * torch.log(p_x_z1 / q_z1) + q_z0 * torch.log(p_x_z0 / q_z0)
    elbo = torch.sum(elbo_x)
    return elbo


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
    gamma_true = 0.8
    thetaA_true = 0.6
    thetaB_true = 0.3
    print("True parameters:")
    print(f"gamma:  {gamma_true:.2f}")
    print(f"thetaA: {thetaA_true:.2f}")
    print(f"thetaB: {thetaB_true:.2f}")
    # Generate data
    data = []
    for i in range(num_samples):
        if np.random.rand() < gamma_true:
            if np.random.rand() < thetaA_true:
                data.append(1)
            else:
                data.append(0)
        else:
            if np.random.rand() < thetaB_true:
                data.append(1)
            else:
                data.append(0)
    data = torch.tensor(data, dtype=torch.int64, requires_grad=False)

    # Define parameters
    gamma = nn.Parameter(torch.randn(1), requires_grad=True)
    thetaA = nn.Parameter(torch.randn(1), requires_grad=True)
    thetaB = nn.Parameter(torch.randn(1), requires_grad=True)

    # Define optimizer
    optimizer = torch.optim.Adam([gamma, thetaA, thetaB], lr=lr)

    # Train model
    loss_list = []
    gamma_list = [F.sigmoid(gamma).item()]
    thetaA_list = [F.sigmoid(thetaA).item()]
    thetaB_list = [F.sigmoid(thetaB).item()]
    for _ in range(num_epochs):
        # Train
        optimizer.zero_grad()
        #loss = -likelihood(data, F.sigmoid(gamma), F.sigmoid(thetaA), F.sigmoid(thetaB)) / num_samples
        #loss = -log_likelihood(data, F.sigmoid(gamma), F.sigmoid(thetaA), F.sigmoid(thetaB)) / num_samples
        #loss = -log_likelihood_with_elbo_and_bad_q(data, F.sigmoid(gamma), F.sigmoid(thetaA), F.sigmoid(thetaB)) / num_samples
        loss = -log_likelihood_with_elbo_and_good_q(data, F.sigmoid(gamma), F.sigmoid(thetaA), F.sigmoid(thetaB)) / num_samples
        loss.backward()
        optimizer.step()
        # Save likelihood and parameters
        loss_list.append(loss.item())
        gamma_list.append(F.sigmoid(gamma).item())
        thetaA_list.append(F.sigmoid(thetaA).item())
        thetaB_list.append(F.sigmoid(thetaB).item())

    # Print results
    print("Estimated parameters:")
    print(f"gamma:  {F.sigmoid(gamma).item():.2f}")
    print(f"thetaA: {F.sigmoid(thetaA).item():.2f}")
    print(f"thetaB: {F.sigmoid(thetaB).item():.2f}")

    # Generate data from posterior
    data_posterior = []
    for i in range(num_samples):
        if np.random.rand() < F.sigmoid(gamma).data.item():
            if np.random.rand() < F.sigmoid(thetaA).data.item():
                data_posterior.append(1)
            else:
                data_posterior.append(0)
        else:
            if np.random.rand() < F.sigmoid(thetaB).data.item():
                data_posterior.append(1)
            else:
                data_posterior.append(0)

    # Plot data and log-likelihood
    fig, axs = plt.subplots(2,2)
    # Original data
    axs[0,0].hist(data, bins=2, density=True)
    axs[0,0].set_title('Data')
    # Generate data
    axs[1,0].hist(data_posterior, bins=2, density=True)
    axs[1,0].set_title('Generate data')
    # Training
    axs[0,1].plot(loss_list)
    axs[0,1].set_title('Loss')
    # Parameters
    axs[1,1].plot(gamma_list, label=r'$\gamma$')
    axs[1,1].plot(thetaA_list, label=r'$\theta_A$')
    axs[1,1].plot(thetaB_list, label=r'$\theta_B$')
    axs[1,1].legend()
    axs[1,1].set_title('Parameters')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
