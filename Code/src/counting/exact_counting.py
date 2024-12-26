import torch
import numpy as np
from src.counting.exact_counting_utils import ActivationRegion, set_empty_activation_pattern, \
                                    embedding, compute_line_region_intersection, update_linear_maps

def exact_count_2D(layers, images_plane, init_vertices):
    n_in = layers[0].weight.shape[1]

    init_region = ActivationRegion(
        vertices=init_vertices,
        activation_pattern=torch.empty((n_in, 1), dtype=torch.float32),
        linear_map=(torch.eye(n_in), torch.zeros(n_in, 1))
    )
    cut_counter = 0
    no_cut_counter = 0

    regions = [init_region]
    with torch.no_grad():
        for l in range(len(layers)-1):      # Iterate over hidden layers
            if isinstance(layers[l], torch.nn.ReLU):
                continue
            W = layers[l].weight            # TODO: Currently W, b requires grad. Faster if not?
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
                    #input = torch.tensor(vertices, dtype=torch.float32).T
                    preactivation = Wi @ (A @ input + c) + bi
                    sign = torch.sign(preactivation)
                    if torch.unique(sign).shape[0] > 1:     # Region is cut
                        cut_counter += 1
                        intersection  = compute_line_region_intersection(region.convex_hull, line_coeffs=(Wi @ A, (Wi @ c).squeeze() + bi), images_plane=images_plane)
                        if len(intersection) > 0:           # TODO: Necessary? Imo this is useless, the region IS cut.
                            idx_to_remove.append(idx)
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
                    else:                               # Region is not cut
                        no_cut_counter += 1
                        if sign[0] == 1:                    # TODO: Find cleaner way to do this
                            new_activation_pattern = old_activation_pattern.clone()
                            new_activation_pattern[i] = 1
                            region.activation_pattern = new_activation_pattern
                        else:
                            new_activation_pattern = old_activation_pattern.clone()
                            new_activation_pattern[i] = 0
                            region.activation_pattern = new_activation_pattern
                regions = [region for idx, region in enumerate(regions) if idx not in idx_to_remove]   # TODO: Find a cleaner way
                regions.extend(new_regions)

            # Update the linear maps of the regions
            update_linear_maps(regions, W, b)

    return regions