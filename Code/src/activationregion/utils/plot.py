import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def plot_regions_exact(regions, with_points=True, labels=None):
    color_set = [
        "blue", "green", "yellow", "orange", "red", "purple", "cyan", "magenta",
        "brown", "pink", "lime", "teal", "navy", "gold", "silver", "gray",
        "maroon", "olive", "violet", "indigo"
    ]
    fig = plt.figure()
    for idx, region in enumerate(regions):
        vertices = region.convex_hull.points[region.convex_hull.vertices]
        color = random.choice(color_set)
        plt.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=1)

    if with_points:
        x = [0, -1, 0]
        y = [0, 0, -1]
        plt.scatter(x, y, color='white')
        #plt.scatter(0, 0, c='white')
        #plt.scatter(-1, 0, c='white')
        #plt.scatter(0, -1, c='white')
        if labels is not None:
            for i, label in enumerate(labels):
                plt.text(x[i], y[i], label, fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.axis('off')
    plt.tight_layout()
    return fig

def plot_regions_sample(inverse_indices, x1, x2, n_samples, with_points=True):
    """
    Contour plot of the map (x1, x2) -> inverse_indices.
    Useful for plotting the activation regions obtained by the sampling method.
    """
    color_set = [
        "blue", "green", "yellow", "orange", "red", "purple", "cyan", "magenta",
        "brown", "pink", "lime", "teal", "navy", "gold", "silver", "gray",
        "maroon", "olive", "violet", "indigo"
    ]
    unique_values = torch.unique(inverse_indices).tolist()
    color_map = {val: random.choice(color_set) for val in unique_values}

    fig = plt.figure()
    plt.contourf(
        x1.numpy(), x2.numpy(),
        np.array(inverse_indices).reshape(n_samples, n_samples),
        colors=[color_map[val] for val in unique_values],
        levels=len(unique_values),
        extend='neither'
    )
    if with_points:     # Add 3 dots representing the images defining the plane (in the plane parametrization)
        plt.scatter(
            np.array([0, -1, 0]), np.array([0, 0, -1]),
            color='black', marker='o', s=100
        )
    return fig