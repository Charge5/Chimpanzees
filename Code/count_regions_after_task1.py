import torch
from src.counting import *
import sys
import os
import random

ROOT_DIR = os.getcwd()
MAMMOTH_PATH = os.path.join(ROOT_DIR, r'src/mammoth')
sys.path.append(MAMMOTH_PATH)
device = torch.device("cpu")
torch.manual_seed(0)                # Reproducibility
random.seed(0)

"""
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

# Randomly choose 3 images to define the plane
MNIST_PATH = os.path.join(MAMMOTH_PATH, r'data/MNIST')
images_plane = random_mnist_images(3, MNIST_PATH)

# Count the number of regions in the plane going through these 3 images (more precisely, only in the square [-1.5, 0.5]^2 not the entire plane)
plane = plane_samples(images_plane, domain_bounds=[-1.5, 0.5], n_samples=1000)
n_regions, inverse_indices = count_regions(features, plane, return_inverse=True)
print(f"Number of regions: {n_regions}")

# Plot the partition of plane
plot_partition(inverse_indices, images_plane, n_samples=1000, domain_bounds=[-1.5, 0.5])