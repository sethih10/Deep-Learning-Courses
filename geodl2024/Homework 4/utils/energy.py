import torch
import numpy as np
from torch import nn


def compute_energy(curve, norm_func):
    velocities = [curve[i] - curve[i - 1] for i in range(1, len(curve))]
    velocities = torch.stack(velocities)  # (n_points - 1, d)
    norm = norm_func(velocities, curve[:-1])
    energy = (norm**2).sum()
    return energy


def compute_geodesic(
    start, end, norm_func, n_points, n_steps=50000, patience=200, print_every=200
):
    curve = torch.Tensor(np.linspace(start, end, n_points))  # shape (n_points, 2)
    curve = [curve[0]] + [nn.Parameter(x) for x in curve[1:-1]] + [curve[-1]]
    optimizer = torch.optim.LBFGS(
        curve[1:-1], lr=1e-1, max_iter=1000, line_search_fn="strong_wolfe"
    )
    min_loss = None
    min_step = None
    for i in range(n_steps):

        def closure():
            optimizer.zero_grad()
            energy = compute_energy(curve, norm_func)
            energy.backward()
            return energy

        optimizer.step(closure)
        if i % print_every == 0:
            print(f"Iteration {i}, energy: {closure().item()}")
        if min_loss is None or closure().item() < min_loss:
            min_loss = closure().item()
            min_step = i
        if i - min_step > patience:
            print(f"Converged at iteration {i}")
            break
    return curve