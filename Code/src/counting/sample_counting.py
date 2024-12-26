import torch
import torch.nn as nn
from src.counting.sample_counting_utils import plane_samples

def sample_count_2D(layers, images, domain_bounds=[-1,1], n_samples=1000, return_inverse=False):
    """
    Count the number of regions in the plane defined by the three images, images[0], images[1], images[2].
    """
    plane = plane_samples(images, domain_bounds=domain_bounds, n_samples=n_samples, return_xy=False)
    n_px = plane.shape[2]           # number of pixels per image
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
    return n_regions