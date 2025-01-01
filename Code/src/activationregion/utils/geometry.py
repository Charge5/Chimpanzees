import numpy as np
import torch

def embedding(coordinates, images_plane):
    """
    Embedding of the 2D plane going through images_plane[0], images_plane[1], images_plane[2]
    in input space.
    """
    coordinates = torch.tensor(coordinates, dtype=torch.float32)
    return coordinates[0] * (images_plane[0] - images_plane[1]) + coordinates[1] * (images_plane[0] - images_plane[2]) + images_plane[0]

def embedding1D(coordinates, images_plane):
    return images_plane[0] + coordinates * (images_plane[1] - images_plane[0])

def compute_line_segment_intersection(p1, p2, line_coeffs, images_plane, embedding=embedding):
    """
    Segment between two points p1, p2: p1 + t * (p2 - p1) with 0 <= t <= 1.
    Line a * x + b * y + c = 0 with line_coeffs: (a, b, c).
    Output: intersection points if it exists, None otherwise.
    """
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
        return p1 + t * (p2 - p1)
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