#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 04')
    parser.add_argument('--num-samples', '-n', type=int, default=100000, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()

    # Set arguments
    num_samples = args.num_samples
    seed = args.seed
    num_epochs = args.num_epochs

    # Set random seed
    np.random.seed(seed)

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
    data = np.array(data)

    # Define parameters
    # Bad guess: 150 epochs
    phi = 0.5
    mu1 = 0.0
    mu2 = 0.0
    sigma1 = 1.0
    sigma2 = 2.0

    # So-so guess: 80 epochs
    # phi = 0.5
    # mu1 = 15.0
    # mu2 = 15.0
    # sigma1 = 4.0
    # sigma2 = 6.0

    # Good guess: 25 epochs
    # phi = 0.7
    # mu1 = 13.0
    # mu2 = 26.0
    # sigma1 = 5.0
    # sigma2 = 2.0

    # Train model
    elbo_list = []
    phi_list = [phi]
    mu1_list = [mu1]
    mu2_list = [mu2]
    sigma1_list = [sigma1]
    sigma2_list = [sigma2]
    for _ in range(num_epochs):
        # Calculate gradient
        c1 = np.exp(-0.5 * (data - mu1)**2 / sigma1**2) / (2 * np.pi * sigma1**2)**0.5 * phi
        c2 = np.exp(-0.5 * (data - mu2)**2 / sigma2**2) / (2 * np.pi * sigma2**2)**0.5 * (1 - phi)
        c3 = c1 + c2
        q1 = c1 / c3
        q2 = c2 / c3
        # Update parameters
        phi = np.mean(q1)
        mu1 = np.sum(q1 * data) / np.sum(q1)
        mu2 = np.sum(q2 * data) / np.sum(q2)
        sigma1 = np.sqrt(np.sum(q1 * (data - mu1)**2) / np.sum(q1))
        sigma2 = np.sqrt(np.sum(q2 * (data - mu2)**2) / np.sum(q2))
        # Calculate ELBO
        elbo = np.mean(np.log(c3 + 1e-10))
        # Save ELBO and parameters
        elbo_list.append(elbo)
        phi_list.append(phi)
        mu1_list.append(mu1)
        mu2_list.append(mu2)
        sigma1_list.append(sigma1)
        sigma2_list.append(sigma2)

    # Print results
    print(f"phi: {phi:.2f}")
    print(f"mu1: {mu1:.2f}")
    print(f"mu2: {mu2:.2f}")
    print(f"sigma1: {sigma1:.2f}")
    print(f"sigma2: {sigma2:.2f}")

    # Generate data from posterior
    data_posterior = []
    for i in range(num_samples):
        if np.random.rand() < phi:
            data_posterior.append(np.random.normal(mu1, sigma1))
        else:
            data_posterior.append(np.random.normal(mu2, sigma2))

    # Plot data and ELBO
    fig, axs = plt.subplots(2,2, figsize=(8,8))
    # Original data
    axs[0,0].hist(data, bins=50, density=True)
    axs[0,0].set_title('Data')
    # Generate data
    axs[1,0].hist(data_posterior, bins=50, density=True)
    axs[1,0].set_title('Generate data')
    # Loss
    axs[0,1].plot(elbo_list)
    axs[0,1].set_title('ELBO')
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
