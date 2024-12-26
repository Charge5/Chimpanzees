import torch
import sys
import os
import random
import matplotlib.pyplot as plt

ROOT_DIR = os.getcwd()
print(ROOT_DIR)
MAMMOTH_PATH = os.path.join(ROOT_DIR, r'src/mammoth')
sys.path.append(MAMMOTH_PATH)
device = torch.device("cpu")
torch.manual_seed(0)                # Reproducibility
random.seed(0)

"""
Make sure you run this script from /Code directory.
This script is an example of how to count the number of regions of the model trained by a specific
CL algorithm of mammoth. It works only for an MLP backbone model and for the MNIST dataset. When you
run mammoth, the backbone model is saved after each task. This script loads the backbone model after
task1 and counts the number of regions in the plane defined by 3 random images.
"""

# Load the model after training on task 1
RESULTS_PATH = os.path.join(MAMMOTH_PATH, r'data/results/ETH/2024-12-24T14_51_49')
PATH = os.path.join(RESULTS_PATH, r'model_task_1.pt')
model = torch.load(PATH, weights_only=False)
model = model.to(device)
model.eval()                        # ContinualModel object
backbone = model.net                # MammothBackbone object
features = backbone._features       # Sequential object containing the layers

"""
1st method: Exact counting (new way)
"""
from src.counting.exact_counting import exact_count_2D
from src.counting.exact_counting_utils import plot_regions, random_mnist_images

# Randomly choose 3 images to define the plane
MNIST_PATH = os.path.join(MAMMOTH_PATH, r'data/MNIST')
images_plane = random_mnist_images(3, MNIST_PATH)

regions = exact_count_2D(features, images_plane, init_vertices=[[-1.5, -1.5], [-1.5, 0.5], [0.5, 0.5], [0.5, -1.5]])
print(f"Number of regions: {len(regions)}")
fig2 = plot_regions(regions)

"""
2nd method: Sample counting (old way)
"""
from src.counting.sample_counting import sample_count_2D
from src.counting.sample_counting_utils import plot_partition

# Count the number of regions in the plane going through these 3 images (more precisely, only in the square [-1.5, 0.5]^2 not the entire plane)
n_regions, inverse_indices = sample_count_2D(features, images_plane, domain_bounds=[-1.5, 0.5], n_samples=1000, return_inverse=True)
print(f"Number of regions: {n_regions}")
fig1 = plot_partition(inverse_indices, images_plane, n_samples=1000, domain_bounds=[-1.5, 0.5])

# Show the plots of both methods
plt.show()