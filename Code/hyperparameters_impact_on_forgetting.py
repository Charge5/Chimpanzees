'''
This is the script used for the figure of chapter 3.

This script will anaylze (post-process) and plot the  Impact of hyperparameters on forgetting. In order
to use that script you will need to have run the scripts "bulk_run_agem_hiof.py" and "bulk_run_lwfmc_hiof.py"
in order to have generate the two files "results-agem.csv" and "results-lwf_mc.csv".
'''

import  pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import  numpy as np
import os
from pathlib import Path

def get_metrics(results_path):
    df = pd.read_csv(results_path)

    # During one run the seed 10 was measured but not in the other run, it was therefore discarded.
    df = df.query("Seed != 10")

    # Due to timeout or connection errors it's possible that some measurement were run twice
    # we therefore remove the duplicated if any. Before hand we need to drop the date column
    # as this value will be different at every run, even with the same hyper parameters/
    df = df.drop(columns=['Date'])
    df = df.drop_duplicates()

    # Get the unique values of each hyperparameters
    depths = df['Depth'].unique()
    epochs = df['Epochs'].unique()
    widths = df['Width'].unique()

    # Generate Cartesian product to get all possible hyperparameters combinations
    combinations = product(depths, epochs, widths)

    # Initialize results dictionary
    results = {2: {'mean': [], 'std': [], 'f_mean': [], 'f_std': []},
               5: {'mean': [], 'std': [], 'f_mean': [], 'f_std': []},
               10: {'mean': [], 'std': [], 'f_mean': [], 'f_std': []}}

    # Loop through combinations
    for d, e, w in combinations:
        data = df[(df['Depth'] == d) & (df['Epochs'] == e) & (df['Width'] == w)]

        # Compute mean and std for current group
        mean_value = data['Mean'].mean()
        std_value = data['Mean'].std()
        f_mean_value = data['Forgetting'].mean()
        f_std_value = data['Forgetting'].std()

        # print(f"d: {d}, e: {e}, w: {w} -> mean = {mean_value:.2f}, std = {std_value:.2f}")

        # Check for missing seeds
        if len(data['Seed']) != 10:
            print(f"Missing seeds for d: {d}, e: {e}, w: {w}")
            for mean, seed in zip(data['Mean'], data['Seed']):
                print(f"  mean: {mean}, seed: {seed}")

        # Store results by depth
        if d in results:
            results[d]['mean'].append(mean_value)
            results[d]['std'].append(std_value)
            results[d]['f_mean'].append(f_mean_value)
            results[d]['f_std'].append(f_std_value)

    # Convert lists to numpy arrays
    for key in results.keys():
        for metric in results[key]:
            results[key][metric] = np.array(results[key][metric])

    # Return the results and widths (required for the x-axis)
    return results, widths

def generate_plots():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 1 row, 2 columns
    for i,model in enumerate(["LwF-MC","A-GEM"]):
        script_dir = Path(__file__).resolve().parent
        results_dir = os.path.join(script_dir,"Data/impact_of_hyperparameters/")
        if model == "LwF-MC":
            results_path = os.path.join(results_dir,"results-lwf_mc.csv")
        elif model == "A-GEM":
            results_path = os.path.join(results_dir, "results-agem.csv")
        else:
            results_path = ""
            # Throw an error for unknown model
            pass


        results,t = get_metrics(results_path)
        # fig.suptitle(val, fontsize=12)

        mean_2, std_2, f_mean_2, f_std_2 = results[2]['mean'], results[2]['std'], results[2]['f_mean'], results[2]['f_std']
        mean_5, std_5, f_mean_5, f_std_5 = results[5]['mean'], results[5]['std'], results[5]['f_mean'], results[5]['f_std']
        mean_10, std_10, f_mean_10, f_std_10 = results[10]['mean'], results[10]['std'], results[10]['f_mean'], results[10]['f_std']


        model_1 = "depth: 2"
        model_2 = "depth: 5"
        model_3 = "depth: 10"

        axes[i][0].plot(t, mean_2, lw=2, label=model_1, color='blue')
        axes[i][0].plot(t, mean_5, lw=2, label=model_2, color='red')
        axes[i][0].plot(t, mean_10, lw=2, label=model_3, color='green')
        axes[i][0].fill_between(t, mean_2+std_2, mean_2-std_2, facecolor='blue', alpha=0.3)
        axes[i][0].fill_between(t, mean_5+std_5, mean_5-std_5, facecolor='red', alpha=0.3)
        axes[i][0].fill_between(t, mean_10 + std_10, mean_10 - std_10, facecolor='green', alpha=0.3)

        # axes[0].set_title(r'Study of forgetting')
        axes[i][0].legend()
        axes[i][0].set_xlabel('Width')
        axes[i][0].set_ylabel('$A_{5}$ [%]')
        axes[i][0].grid()

        axes[i][0].set_title(f"{model} average accuracy")

        axes[i][1].plot(t, f_mean_2, lw=2, label=model_1, color='blue')
        axes[i][1].plot(t, f_mean_5, lw=2, label=model_2, color='red')
        axes[i][1].plot(t, f_mean_10, lw=2, label=model_3, color='green')
        axes[i][1].fill_between(t, f_mean_2+f_std_2, f_mean_2-f_std_2, facecolor='blue', alpha=0.3)
        axes[i][1].fill_between(t, f_mean_5+f_std_5, f_mean_5-f_std_5, facecolor='red', alpha=0.3)
        axes[i][1].fill_between(t, f_mean_10 + f_std_10, f_mean_10 - f_std_10, facecolor='green', alpha=0.3)

        # axes[1].set_title(r'Study of forgetting')
        # axes[i][1].legend(loc='upper left')
        axes[i][1].legend()
        axes[i][1].set_xlabel('Width')
        axes[i][1].set_ylabel('$F_{5}$ [%]')
        axes[i][1].grid()

        axes[i][1].set_title(f"{model} forgetting")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_plots()