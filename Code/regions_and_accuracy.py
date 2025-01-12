import torch
import sys
import os
import random
import datetime
import time
import logging
import json
from utils_for_experiments import create_dir, setup_logging, run_command, format_time

### ---------------------------------------------------------------- ###
# Make sure to run this script from the /Code directory.

# GOAL:   Run mammoth and save the accuracy of the model on each task 
#         during training.
# OUTPUT: Json file available in the folder 'Code/regions_and_accuracy',
#         containing all computed accuracies.

# DETAILS: Use mammoth to train an MLP model on sequential MNIST using
# LwF-MC algorithm. During training, the accuracy of the model
# on each previously seen task at different epochs is saved.

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
dir_path = create_dir(dir_path=r'regions_and_accuracy', format='%Y-%m-%d')

# Set the parameters of the experiment
# You can change edit them directly in the dictionary
params = {
    'n_experiment': 10,    # Number of independent runs
    'lr': 0.01,
    'n_epochs': 50,
    'mlp_hidden_size': 70,
    'mlp_hidden_depth': 2,
    'info': {}
}

# Run the experiment
width = params['mlp_hidden_size']
depth = params['mlp_hidden_depth']
begin_time = datetime.datetime.now().isoformat()
begin_time = format_time(begin_time)

setup_logging(log_path=dir_path, log_name=f'log_{depth}_{width}_{begin_time}.txt')
logging.info(f"Results will be saved at: {dir_path}")
logging.info(f"Starting the experiment with parameters:\n {params}")

params['model'] = 'lwf-mc'
params['info']['count_vals'] = [0, 1, 2, 5, 10, 25, 40, 45, 48, 49]     # Epochs at which the accuracies are saved
params['info']['n_tasks'] = 5
params['info']['seeds'] = [i for i in range(params['n_experiment'])]
params['info']['RESULTS_PATH'] = []

tic = time.time()
acc = {}
for k in range(params['n_experiment']):
    logging.info(f'Experiment: {k+1}')
    seeds = params['info']['seeds']
    logging.info(f'Seed: {seeds[k]}')

    cmd = f"python utils/main.py --dataset seq-mnist --backbone mnistmlp --model {params['model']}  \
            --lr {params['lr']} --seed {params['info']['seeds'][k]} --n_epochs {params['n_epochs']}\
            --mlp_hidden_size {params['mlp_hidden_size']} --mlp_hidden_depth {params['mlp_hidden_depth']}\
            --save_accuracy_within_tasks True"
    logging.info(f'Running mammoth with command:\n {cmd}')
    os.chdir(MAMMOTH_PATH)
    current_time = datetime.datetime.now()
    if current_time.second > 50:    # Wait a few seconds, to get the correct filename
        time.sleep(11)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    current_time = current_time.replace(":", "_").split(".")[0]
    run_command(cmd)    # Run mammoth
    os.chdir(ROOT_DIR)

    RESULTS_PATH = os.path.join(MAMMOTH_PATH, f'data/results/ETH/{current_time}')
    logging.info(f'Mammoth results of experiment {k+1} saved at: {RESULTS_PATH}')
    params['info']['RESULTS_PATH'].append(RESULTS_PATH)

    PATH = os.path.join(RESULTS_PATH, f'accuracy.json')
    with open(PATH, "r") as file:
        accuracy = json.load(file)
    logging.info(f'Accuracy: {accuracy}')
    acc[f"exp_{k}"] = accuracy

logging.info('Experiment finished!')

logging.info('Saving results...')
# Save all the computed accuracies for all independent runs
with open(dir_path + f"/accuracy_{depth}_{width}_{begin_time}.json", "w") as f:
    json.dump(acc, f, indent=4)

toc = time.time()
logging.info(f'Duration: {toc - tic} seconds.')

### ------------------------ EXPERIMENT DONE ---------------------------- ###