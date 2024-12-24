import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms

def plane_samples(images, domain_bounds=[-1, 1], n_samples=500, return_xy=False):
    """ 
    Define the plane going through the three images images[0], images[1], images[2].
    Sample n_samples points on the plane, within the domain bounds.
    return_xy: only useful for the plot later.
    """
    image_size = int(np.sqrt(images[0].shape[0]))
    x1 = torch.linspace(domain_bounds[0], domain_bounds[1], n_samples)
    x2 = torch.linspace(domain_bounds[0], domain_bounds[1], n_samples)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='xy')

    basis = torch.concatenate((images[0] - images[1], images[0] - images[2]), dim=1).T
    plane = grid_x1.reshape((n_samples, n_samples, 1)) * basis[0] + grid_x2.reshape((n_samples, n_samples, 1)) * basis[1]
    plane = plane + images[0].view(-1, image_size * image_size)
    if return_xy:
        return plane, x1, x2
    else:
        return plane

def count_regions(layers, plane, return_inverse=True):
    """
    Count the number of regions in the plane defined by the three images.
    return_inverse: only useful for the plot later.
    """
    n_px = plane.shape[2]           # number of pixels per image
    n_samples = plane.shape[0]
    x = plane.reshape((n_samples * n_samples, n_px))
    relu = nn.ReLU()
    activations = []
    for l in range(len(layers)):
        if isinstance(layers[l], nn.Linear):
            with torch.no_grad():
                x = layers[l](x)
                x = relu(x)
                activations.append( (x>0).squeeze().int() )
    activations = torch.cat(activations, dim=1)
    
    unique_activations, inverse_indices = torch.unique(activations, dim=0, return_inverse=True)
    n_regions = len(unique_activations)
    if return_inverse:
        return n_regions, inverse_indices
    else:
        return n_regions
    
def plot_partition(inverse_indices, images, n_samples, domain_bounds=[-1, 1]):
    color_set = [
        "blue", "green", "yellow", "orange", "red", "purple", "cyan", "magenta",
        "brown", "pink", "lime", "teal", "navy", "gold", "silver", "gray",
        "maroon", "olive", "violet", "indigo"
    ]
    plane, x1, x2 = plane_samples(images, n_samples=n_samples, domain_bounds=domain_bounds, return_xy=True)
    unique_values = torch.unique(inverse_indices).tolist()
    color_map = {val: random.choice(color_set) for val in unique_values}

    plt.contourf(
        x1.numpy(), x2.numpy(),
        np.array(inverse_indices).reshape(n_samples, n_samples),
        colors=[color_map[val] for val in unique_values],
        levels=len(unique_values),
        extend='neither'
    )
    plt.scatter(
        np.array([0, -1, 0]), np.array([0, 0, -1]),
        color='black', marker='o', s=100
    )
    plt.show()

def random_mnist_images(n, PATH):
    # Load dataset and select three images from MNIST
    image_size = 28
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(image_size * image_size, 1)),
    ])
    train_dataset = datasets.MNIST(PATH,
                            train=True, download=True, transform=transform)
    
    idx = [random.randint(0, len(train_dataset)-1) for i in range(n)]
    images = []
    for i in range(len(idx)):
        images.append(train_dataset[i][0])
    images = torch.cat(images, dim=1).T
    images = images[..., None]
    return images