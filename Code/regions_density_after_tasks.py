import torch
import sys
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from src.activationregion.core import exact_count_2D, count_for_experiment
from src.activationregion.utils.plot import plot_regions_exact
import datetime
import time
import subprocess
import logging
from torchvision import datasets, transforms
from utils_for_experiments import create_dir, setup_logging, save_params, run_command, format_time

### ---------------------------------------------------------------- ###
# Make sure to run this script from the /Code directory.

# GOAL: Measure the evolution of the number of activation regions during 
#       continual learning.
# OUTPUT: Results, logs and plot will be saved in the folder 'Code/counting_regions_during_CL'.

# DETAILS: This script runs mammoth on an MLP model. Mammoth will save the model at different
# epochs during training. We then load the saved models and count the number of activation
# regions.

### ------------- UTILS, SEE BELOW FOR THE EXPERIMENT -------------- ###

ROOT_DIR = os.getcwd()
MAMMOTH_PATH = os.path.join(ROOT_DIR, r'src/mammoth')
MNIST_PATH = os.path.join(MAMMOTH_PATH, r'data/MNIST')
sys.path.append(MAMMOTH_PATH)
device = torch.device("cpu")
torch.manual_seed(0)                # Reproducibility
random.seed(0)

### ------------------------ EXPERIMENT ---------------------------- ###

# Create a directory to save results and logs
dir_path = create_dir(dir_path=r'regions_density_after_tasks', format='%Y-%m-%d')

begin_time = datetime.datetime.now().isoformat()
begin_time = format_time(begin_time)
setup_logging(log_path=dir_path, log_name=f'log_{2}_{70}_{begin_time}.txt')
logging.info(f"Results will be saved at: {dir_path}")
logging.info(f"Starting the experiment.")

# Load the MNIST dataset
logging.info('Loading MNIST dataset...')
image_size = 28
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(image_size * image_size, 1)),
])
train_dataset = datasets.MNIST(MNIST_PATH, train=True, download=True, transform=transform)
# Randomly select 3 images, respectively with labels 0, 2 and 4
logging.info('Selecting 3 images ...')
labels = [0, 2, 4]
images = []
for label in labels:
    indices = np.where(train_dataset.targets == label)[0]
    index = random.choice(indices)
    image, _ = train_dataset[index]
    images.append(image)
images = torch.cat(images, dim=1).T
images = images[..., None]

# Load the models corresponding to the run of mammoth on MLP depth 2 width 70 using LwF-MC
n_tasks = 5
init_vertices=[[-1.5, -1.5], [-1.5, 0.5], [0.5, 0.5], [0.5, -1.5]]
for i in range(1, 4):  # Only load the models after the first 3 tasks
    RESULTS_PATH = os.path.join(MAMMOTH_PATH, r'data/results/ETH/2025-01-03 13_38')
    PATH = os.path.join(RESULTS_PATH, f'model_task_{i+1}_epoch_{0}.pt')
    logging.info(f'Loading model from: {PATH}')
    model = torch.load(PATH, weights_only=False)
    model = model.to(device)
    model.eval()
    backbone = model.net
    features = backbone._features

    logging.info(f'Determining the regions ...')
    regions = exact_count_2D(features, images, init_vertices=init_vertices)
    logging.info(f'Number of regions: {len(regions)}')
    fig = plot_regions_exact(regions, with_points=True, labels=labels)

    os.chdir(ROOT_DIR)
    fig.savefig(os.path.join(dir_path, f'regions_task_{i+1}_epoch_{0}.png'), bbox_inches='tight', pad_inches=0)
    logging.info(f'Plot saved at: {dir_path}')