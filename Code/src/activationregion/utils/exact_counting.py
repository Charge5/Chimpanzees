import torch
from scipy.spatial import ConvexHull

# Definition of main object: the region
class ActivationRegion():
    def __init__(self, vertices, activation_pattern, linear_map):
        self.convex_hull = ConvexHull(vertices)
        self.activation_pattern = activation_pattern
        self.linear_map = linear_map

    def update_linear_map(self, W, b):
        A, c = self.linear_map
        a = self.activation_pattern
        A_new = a * W @ A
        c_new = a * W @ c + a * b.view(-1, 1)
        self.linear_map = (A_new, c_new)

    def update_activation_pattern(self, idx, sign):
        if sign == 1:
            self.activation_pattern[idx] = 1
        else:
            self.activation_pattern[idx] = 0

# Utils for the counting
def set_empty_activation_pattern(regions, n_neurons):
    for region in regions:
        region.activation_pattern = torch.empty((n_neurons, 1), dtype=torch.float32)

def update_linear_maps(regions, W, b):
    for region in regions:
        region.update_linear_map(W, b)