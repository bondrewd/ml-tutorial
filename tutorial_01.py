#!/usr/bin/env python3

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tutorial 01')
    parser.add_argument('--num-samples', '-n', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs')
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

    # Define parameters
    # phi = torch.tensor(0.0, requires_grad=True)
    # p1 = torch.tensor(0.0, requires_grad=True)
    # p2 = torch.tensor(0.0, requires_grad=True)
    # phi = torch.tensor(2.0, requires_grad=True)
    # p1 = torch.tensor(1.0, requires_grad=True)
    # p2 = torch.tensor(-1.0, requires_grad=True)
    phi = torch.tensor(-3.0, requires_grad=True)
    p1 = torch.tensor(-2.0, requires_grad=True)
    p2 = torch.tensor(3.0, requires_grad=True)
    # phi = torch.tensor(-np.log((1/phi_true)-1)+0.1, requires_grad=True)
    # p1 = torch.tensor(-np.log((1/p1_true)-1)+0.1, requires_grad=True)
    # p2 = torch.tensor(-np.log((1/p2_true)-1)+0.1, requires_grad=True)

    # Train model
    elbo_list = []
    phi_list = [torch.sigmoid(phi).data.item()]
    p1_list = [torch.sigmoid(p1).data.item()]
    p2_list = [torch.sigmoid(p2).data.item()]
    for _ in range(num_epochs):
        # Reset gradients
        phi.grad = None
        p1.grad = None
        p2.grad = None
        # Calculate probabilities
        phi_sig = torch.sigmoid(phi)
        p1_sig = torch.sigmoid(p1)
        p2_sig = torch.sigmoid(p2)
        # Calculate ELBO
        elbo = torch.zeros(1)
        for x in data:
            c1 = phi_sig * (p1_sig*x + (1-p1_sig)*(1-x))
            c2 = (1-phi_sig) * (p2_sig*x + (1-p2_sig)*(1-x))
            elbo += torch.log(c1 + c2)
        elbo /= len(data)
        # Calculate gradients
        elbo.backward()
        # Update parameters
        phi.data += lr * phi.grad
        p1.data += lr * p1.grad
        p2.data += lr * p2.grad
        # Save ELBO and parameters
        elbo_list.append(elbo.item())
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
        if np.random.rand() < torch.sigmoid(phi).data.item():
            if np.random.rand() < torch.sigmoid(p1).data.item():
                data_posterior.append(1)
            else:
                data_posterior.append(0)
        else:
            if np.random.rand() < torch.sigmoid(p2).data.item():
                data_posterior.append(1)
            else:
                data_posterior.append(0)

    # Plot data and ELBO
    fig, axs = plt.subplots(2,2)
    axs[0,0].hist(data, bins=2)
    axs[0,1].plot(elbo_list)
    axs[1,0].plot(phi_list, label='phi')
    axs[1,0].plot(p1_list, label='p1')
    axs[1,0].plot(p2_list, label='p2')
    axs[1,0].legend()
    axs[1,1].hist(data_posterior, bins=2)
    plt.show()

if __name__ == '__main__':
    main()
