## ML Tutorials

This repository contains tutorials on generative models.
All the tutorials are written in Python and use PyTorch.
To execute the tutorials, clone the repository and run them using uv.

```bash
git clone https://github.com/bondrewd/ml-tutorial.git
cd ml-tutorial
uv sync
uv run tutorial_XX.py
```

where you need to replace *XX* with the tutorial number.

## Tutorials
The following tutorials are available:

1. Log-likelihood fitting of a biassed coin toss by maximizing the ELBO
2. Log-likelihood fitting of a mixture of Gaussian distributions by maximizing the ELBO
3. Log-likelihood fitting of a mixture of Gaussian distributions using an analytical solution
4. Log-likelihood fitting of a mixture of Gaussian distributions using a VAE
5. Log-likelihood fitting of the MNIST dataset using a VAE
