import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import QhullError
from src.activationregion.utils.sample_counting import plane_samples
from src.activationregion.utils.exact_counting import ActivationRegion, set_empty_activation_pattern, \
                                                    update_linear_maps
from src.activationregion.utils.geometry import embedding, compute_line_region_intersection
from src.activationregion.utils.utils import images_for_experiment

def sample_count_2D(layers, images, domain_bounds=[-1,1], n_samples=1000, return_inverse=False):
    """
    Sampling method to count the number of activation.
    Counting restricted to a bounded domain in plane going through 3 images, images[0], images[1], images[2].
    """
    if return_inverse:
        plane, x1, x2 = plane_samples(images, domain_bounds=domain_bounds, n_samples=n_samples, return_xy=True)
    else:
        plane = plane_samples(images, domain_bounds=domain_bounds, n_samples=n_samples, return_xy=False)
    n_px = plane.shape[2]
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
        return n_regions, inverse_indices, x1, x2
    return n_regions

def exact_count_2D(layers, images_plane, init_vertices):
    n_in = layers[0].weight.shape[1]

    init_region = ActivationRegion(
        vertices=init_vertices,
        activation_pattern=torch.empty((n_in, 1), dtype=torch.float32),
        linear_map=(torch.eye(n_in), torch.zeros(n_in, 1))
    )

    regions = [init_region]
    with torch.no_grad():
        for l in range(len(layers)):      # Iterate over hidden layers
            if isinstance(layers[l], torch.nn.ReLU):
                continue
            W = layers[l].weight
            b = layers[l].bias
            n_neurons = W.shape[0]
            set_empty_activation_pattern(regions, n_neurons)
            for i in range(n_neurons):      # Iterate over neurons of current layer
                Wi = W[i]
                bi = b[i]
                new_regions = []
                idx_to_remove = []
                for idx, region in enumerate(regions):      # Iterate over regions
                    old_activation_pattern = region.activation_pattern
                    A, c = region.linear_map
                    vertices = region.convex_hull.points[region.convex_hull.vertices]
                    input = embedding(vertices.T, images_plane)
                    preactivation = Wi @ (A @ input + c) + bi
                    sign = torch.sign(preactivation)
                    if torch.unique(sign).shape[0] > 1:     # Region is cut
                        try:
                            intersection  = compute_line_region_intersection(region.convex_hull, line_coeffs=(Wi @ A, (Wi @ c).squeeze() + bi), images_plane=images_plane)
                            if len(intersection) <= 1:          # Intersection is a single point
                                region.update_activation_pattern(i, sign[0])
                            #elif len(intersection) >= 0:           # TODO: Necessary? Imo this is useless, the region IS cut.
                            else:
                                new_vertices = np.concatenate((vertices[np.where(sign > 0)[0]], intersection), axis=0)
                                new_activation_pattern = old_activation_pattern.clone()
                                new_activation_pattern[i] = 1
                                new_region = ActivationRegion(
                                    vertices=new_vertices,
                                    activation_pattern=new_activation_pattern,
                                    linear_map=(A, c)
                                )
                                new_regions.append(new_region)

                                new_vertices = np.concatenate((vertices[np.where(sign < 0)[0]], intersection), axis=0)
                                new_activation_pattern = old_activation_pattern.clone()
                                new_activation_pattern[i] = 0
                                new_region = ActivationRegion(
                                    vertices=new_vertices,
                                    activation_pattern=new_activation_pattern,
                                    linear_map=(A, c)
                                )
                                new_regions.append(new_region)

                                idx_to_remove.append(idx)
                        except QhullError:
                            region.update_activation_pattern(i, sign[0])
                    else:                               # Region is not cut
                        region.update_activation_pattern(i, sign[0])
                regions = [region for idx, region in enumerate(regions) if idx not in idx_to_remove]   # TODO: Find a cleaner way
                regions.extend(new_regions)

            # Update the linear maps of the regions
            update_linear_maps(regions, W, b)

    return regions

def count_for_experiment(layers, init_vertices, MNIST_PATH, image_size=28, n_planes=5):
    image_set = images_for_experiment(n_planes, MNIST_PATH, image_size=image_size)
    counts = np.zeros((len(image_set),))
    for i, images_plane in enumerate(image_set):
        regions = exact_count_2D(layers, images_plane, init_vertices)
        counts[i] = len(regions)
    return np.mean(counts)