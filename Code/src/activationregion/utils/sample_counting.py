import torch
import numpy as np

def plane_samples(points, domain_bounds=[-1, 1], n_samples=500, return_xy=False):
    """ 
    Sample the plane going through the three points p0, p1, p2.

    The plane is parametrized by {x1 * (p0 - p1) + x2 * (p0 - p2) + p0 | x1, x2 in R}.
    Return n_samples**2 samples on uniform grid within the domain bounds.
    return_xy: return the samples, useful for the plotting.
    """
    image_size = int(np.sqrt(points[0].shape[0]))
    x1 = torch.linspace(domain_bounds[0], domain_bounds[1], n_samples)
    x2 = torch.linspace(domain_bounds[0], domain_bounds[1], n_samples)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='xy')

    basis = torch.concatenate((points[0] - points[1], points[0] - points[2]), dim=1).T
    plane = grid_x1.reshape((n_samples, n_samples, 1)) * basis[0] + grid_x2.reshape((n_samples, n_samples, 1)) * basis[1]
    plane = plane + points[0].view(-1, image_size * image_size)
    if return_xy:
        return plane, x1, x2
    else:
        return plane