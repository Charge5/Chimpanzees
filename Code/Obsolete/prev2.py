import os
import json
import torch
import sys
import random
from src.activationregion.core import exact_count_2D
from src.activationregion.utils.utils import random_mnist_images
import time


root_dir = os.getcwd()
# Get the results directory
settings_path = os.path.join(root_dir,"src/settings.json")
MAMMOTH_PATH = os.path.join(root_dir, r'src/mammoth')
sys.path.append(MAMMOTH_PATH)
device = torch.device("cpu")
torch.manual_seed(0)                # Reproducibility
random.seed(0)

f = open(settings_path)
data = f.read()
data = json.loads(data)
for setting in data:
    print(setting)
    start_time = time.time()  # Start timing
    RESULTS_PATH = os.path.join(MAMMOTH_PATH, f'data/results/ETH/{setting["path"]}')
    PATH = os.path.join(RESULTS_PATH, r'model_task_5.pt')
    model = torch.load(PATH, weights_only=False)
    model = model.to(device)
    model.eval()  # ContinualModel object
    backbone = model.net  # MammothBackbone object
    features = backbone._features  # Sequential object containing the layers

    # Randomly choose 3 images to define the plane
    images_plane = random_mnist_images(3)

    # Count the number of regions in the plane going through these 3 images
    # (more precisely, only in the square [-1.5, 0.5]^2, not the entire plane)
    regions = exact_count_2D(features, images_plane, init_vertices=[[-1.5, -1.5], [-1.5, 0.5], [0.5, 0.5], [0.5, -1.5]])
    number_of_regions = len(regions)
    # print(f"Number of regions: {number_of_regions}")
    setting["regions"]=number_of_regions
    # print(PATH)
    # print(setting)
    end_time = time.time()  # Start timing
    elapsed_time = end_time - start_time
    print(f"Time for iteration : {elapsed_time:.4f} seconds\n")


    import json
    data_str = json.dumps(data)
    f = open("data2.json","w")
    f.write(data_str)
    f.close()

# # Load the model after training on task 1
# RESULTS_PATH = os.path.join(MAMMOTH_PATH, r'data/results/ETH/2024-12-24T14_51_49')
# PATH = os.path.join(RESULTS_PATH, r'model_task_5.pt')
# model = torch.load(PATH, weights_only=False)
# model = model.to(device)
# model.eval()                        # ContinualModel object
# backbone = model.net                # MammothBackbone object
# features = backbone._features       # Sequential object containing the layers
#
# # Randomly choose 3 images to define the plane
# images_plane = random_mnist_images(3)
#
# # Count the number of regions in the plane going through these 3 images
# # (more precisely, only in the square [-1.5, 0.5]^2, not the entire plane)
# # regions = exact_count_2D(features, images_plane, init_vertices=[[-1.5, -1.5], [-1.5, 0.5], [0.5, 0.5], [0.5, -1.5]])
# # number_of_regions = len(regions)
# # print(f"Number of regions: {number_of_regions}")


