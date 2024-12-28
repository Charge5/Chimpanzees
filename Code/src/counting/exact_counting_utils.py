import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import random
from torchvision import datasets, transforms

def embedding(coordinates, images_plane):
    """
    Embedding of the 2D plane going through images_plane[0], images_plane[1], images_plane[2]
    in input space.
    """
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    return coordinates[0] * (images_plane[0] - images_plane[1]) + coordinates[1] * (images_plane[0] - images_plane[2]) + images_plane[0]

# Geometric utils
def compute_line_segment_intersection(p1, p2, line_coeffs, images_plane):
    """
    Segment between two points p1, p2: p1 + t * (p2 - p1) with 0 <= t <= 1.
    Line a * x + b * y + c = 0 with line_coeffs: (a, b, c).
    Output: intersection points if it exists, None otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    p1_emb = embedding(p1[..., np.newaxis], images_plane)
    p2_emb = embedding(p2[..., np.newaxis], images_plane)

    w = line_coeffs[0]
    b = line_coeffs[1]

    denominator = w @ (p2_emb - p1_emb)
    if abs(denominator) < 1e-9:  # Parallel line    TODO: Check if the segment is on the line?
        return None

    t = -(b + w @ p1_emb) / denominator
    t = t.item()
    if 0 < t < 1:              # Check if intersection is within the segment
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None

def compute_line_region_intersection(convex_hull, line_coeffs, images_plane):
    """
    convex_hull: scipy.spatial.ConvexHull object.
    line_coeffs: (a, b, c), defining the line a * x + b * y + c = 0.
    Output: list of intersection points.
    """
    intersections = []
    points = convex_hull.points
    for edge in convex_hull.simplices:
        p1, p2 = points[edge[0]], points[edge[1]]
        point = compute_line_segment_intersection(p1, p2, line_coeffs, images_plane)
        if point is not None:
            intersections.append(point)
    return intersections

# Definition of main object: the region
class ActivationRegion():
    def __init__(self, vertices, activation_pattern, linear_map):
        self.convex_hull = ConvexHull(vertices)     # TODO: Might be an overkill? Just need the vertices
        self.activation_pattern = activation_pattern
        self.linear_map = linear_map

    def update_linear_map(self, W, b):
        A, c = self.linear_map
        a = self.activation_pattern
        A_new = a * W @ A
        c_new = a * W @ c + a * b.view(-1, 1)
        self.linear_map = (A_new, c_new)

# Utils for the counting
def set_empty_activation_pattern(regions, n_neurons):
    for region in regions:
        region.activation_pattern = torch.empty((n_neurons, 1), dtype=torch.float32)

def update_linear_maps(regions, W, b):
    for region in regions:
        region.update_linear_map(W, b)

def random_mnist_images(n, PATH, image_size=28):
    """
    Randomly select n images from MNIST training set located at PATH.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size,image_size)),
        transforms.Lambda(lambda x: x.view(image_size * image_size, 1)),
    ])
    train_dataset = datasets.MNIST(PATH,
                            train=True, download=True, transform=transform)
    
    idx = [random.randint(0, len(train_dataset)-1) for i in range(n)]
    images = []
    for i in range(len(idx)):
        images.append(train_dataset[idx[i]][0])
    images = torch.cat(images, dim=1).T
    images = images[..., None]
    return images

def plot_regions(regions):
    color_set = [
        "blue", "green", "yellow", "orange", "red", "purple", "cyan", "magenta",
        "brown", "pink", "lime", "teal", "navy", "gold", "silver", "gray",
        "maroon", "olive", "violet", "indigo"
    ]
    fig = plt.figure()
    # Color each region with a ranom color
    for idx, region in enumerate(regions):
        vertices = region.convex_hull.points[region.convex_hull.vertices]
        color = random.choice(color_set)
        plt.fill(vertices[:, 0], vertices[:, 1], color=color, alpha=1)

    plt.scatter(0, 0, c='black')
    plt.scatter(-1, 0, c='black')
    plt.scatter(0, -1, c='black')
    return fig