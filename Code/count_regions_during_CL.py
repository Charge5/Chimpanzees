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
from utils_for_experiments import create_dir, setup_logging, save_params, run_command, format_time

### ------------- UTILS, SEE BELOW FOR THE EXPERIMENT -------------- ###

ROOT_DIR = os.getcwd()
MAMMOTH_PATH = os.path.join(ROOT_DIR, r'src/mammoth')
MNIST_PATH = os.path.join(MAMMOTH_PATH, r'data/MNIST')
sys.path.append(MAMMOTH_PATH)
device = torch.device("cpu")
torch.manual_seed(0)                # Reproducibility
random.seed(0)

def save_plot(values, results, n_tasks, save_path, filename):
    colors = plt.cm.Set1(np.linspace(0, 1, n_tasks))

    mean_curve = np.mean(results, axis=2)
    std_dev = np.std(results, axis=2)

    x_concat = []
    y_concat = []
    for i in range(n_tasks):
        x_concat.extend(np.array(values) + i * max(values))
        y_concat.extend(mean_curve[i])

    plt.figure(figsize=(12, 6))
    for i in range(n_tasks):
        x_task = np.array(values) + i * max(values)
        y_task = mean_curve[i]
        std_dev_task = std_dev[i]
        plt.plot(x_task, y_task, marker='.', markersize=4, color=colors[i], linestyle='-', linewidth=0.8, label=f'Task {i + 1}')
        plt.fill_between(x_task, y_task - std_dev_task, y_task + std_dev_task, color=colors[i], alpha=0.3)

        if i < n_tasks - 1:
            next_x = values[0] + (i + 1) * max(values)
            next_y = mean_curve[i + 1][0]
            plt.plot([x_task[-1], next_x], [y_task[-1], next_y], color=colors[i + 1], linestyle='-')

    for i in range(n_tasks):
        task_label_position = np.mean(np.array(values) + i * max(values))
        plt.text(task_label_position,
                 plt.ylim()[0] + 0.04 * (plt.ylim()[1] - plt.ylim()[0]),
                 f"Task {i + 1}",
                 ha='center',
                 va='top',
                 fontsize=15,
                 color=colors[i])

    plt.xlabel('Epochs and tasks', fontsize=20)
    plt.ylabel('Number of regions', fontsize=20)
    custom_xticks = [0]
    custom_xtick_labels = ["0"]

    for i in range(n_tasks):
        xticks_for_task = np.array([20, 40, 60]) + i * max(values)
        custom_xticks.extend(xticks_for_task)
        if i == 0:
            custom_xtick_labels.extend([str(tick) for tick in [20, 40, 60]])
        else:
            custom_xtick_labels.extend([str(tick) for tick in [20, 40, 60]])
    plt.xticks(ticks=custom_xticks, labels=custom_xtick_labels, rotation=45, ha='right', fontsize=15)
    plt.grid(alpha=0.3)
    plt.yticks(fontsize=15)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))

### ------------- EXPERIMENT -------------- ###

dir_path = create_dir(dir_path=r'counting_regions_during_CL', format='%Y-%m-%d')

params = {
    'n_experiment': 10,
    'n_planes': 5,
    'init_vertices': [[-500, -500], [-500, 500], [500, 500], [500, -500]],
    'lr': 0.01,
    'n_epochs': 60,
    'mlp_hidden_size': 20,
    'mlp_hidden_depth': 2,
    'info': {}
}
width = params['mlp_hidden_size']
depth = params['mlp_hidden_depth']
begin_time = datetime.datetime.now().isoformat()
begin_time = format_time(begin_time)

setup_logging(log_path=dir_path, log_name=f'log_{depth}_{width}_{begin_time}.txt')
logging.info(f"Results will be saved at: {dir_path}")
logging.info(f"Starting the experiment with parameters:\n {params}")

params['info']['count_vals'] = [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 57, 58, 59]
params['info']['n_tasks'] = 5
params['info']['seeds'] = [i for i in range(params['n_experiment'])]
params['info']['RESULTS_PATH'] = []

regions_evol = np.zeros((params['info']['n_tasks'], len(params['info']['count_vals']), params['n_experiment']))
tic = time.time()
for k in range(params['n_experiment']):
    logging.info(f'Experiment: {k+1}')
    seeds = params['info']['seeds']
    logging.info(f'Seed: {seeds[k]}')

    cmd = f"python utils/main.py --dataset seq-mnist --backbone mnistmlp --model lwf-mc  \
            --lr {params['lr']} --seed {params['info']['seeds'][k]} --n_epochs {params['n_epochs']}\
            --mlp_hidden_size {params['mlp_hidden_size']} --mlp_hidden_depth {params['mlp_hidden_depth']}\
            --save_models_within_tasks True"
    logging.info(f'Running mammoth with command:\n {cmd}')
    os.chdir(MAMMOTH_PATH)
    current_time = datetime.datetime.now()
    if current_time.second > 50:    # Wait a few seconds, to get the correct filename
        time.sleep(11)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    current_time = current_time.replace(":", "_").split(".")[0]
    run_command(cmd)
    os.chdir(ROOT_DIR)

    RESULTS_PATH = os.path.join(MAMMOTH_PATH, f'data/results/ETH/{current_time}')
    logging.info(f'Mammoth results of experiment {k+1} saved at: {RESULTS_PATH}')
    params['info']['RESULTS_PATH'].append(RESULTS_PATH)

    for i in range(params['info']['n_tasks']):
        logging.info(f'Task: {i+1}')
        for j in range(len(params['info']['count_vals'])):
            epoch = params['info']['count_vals'][j]
            logging.info(f'Epoch: {epoch}')
            PATH = os.path.join(RESULTS_PATH, f'model_task_{i+1}_epoch_{epoch}.pt')
            logging.info(f'Loading model from: {PATH}')
            model = torch.load(PATH, weights_only=False)
            model = model.to(device)
            model.eval()
            backbone = model.net
            features = backbone._features

            logging.info('Counting regions...')
            n_regions = count_for_experiment(features, init_vertices=params['init_vertices'],
                                            MNIST_PATH=MNIST_PATH, n_planes=params['n_planes'])
            regions_evol[i, j, k] = n_regions

logging.info('Experiment finished!')

logging.info('Saving results...')
os.chdir(ROOT_DIR)
depth = params['mlp_hidden_depth']
width = params['mlp_hidden_size']
np.save(os.path.join(dir_path, f'results_{depth}_{width}_{begin_time}'), regions_evol)
save_plot(params['info']['count_vals'], regions_evol, params['info']['n_tasks'], save_path=dir_path, filename=f'plot_{depth}_{width}_{begin_time}.png')
save_params(params, dir_path, filename=f'parameters_{depth}_{width}_{begin_time}.txt')

toc = time.time()
logging.info(f'Duration: {toc - tic} seconds.')