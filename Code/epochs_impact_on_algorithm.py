'''
This is the script used for the figure of Appendix C. Study of the number of epochs on the
continuous algorithm .

This script will anaylze (post-process) and plot the  Impact of hyperparameters on forgetting. In order
to use that script you will need to have run the scripts "bulk_run_agem_hiof.py" and "bulk_run_lwfmc_hiof"
in order to have generate the two files "results-agem.csv" and "results-lwf_mc.csv".
'''

import  pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import  numpy as np
import os
from pathlib import Path
import ast

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



    # Return the results and widths (required for the x-axis)
    return results, widths

def generate_plots_classes():
    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 1 row, 2 columns
    for i,model in enumerate(["LwF-MC","A-GEM"]):
    # for i, model in enumerate(["A-GEM"]):
        script_dir = Path(__file__).resolve().parent
        results_dir = os.path.join(script_dir,"Data/impact_of_hyperparameters/")
        if model == "LwF-MC":
            results_path = os.path.join(results_dir,"results-lwf_mc-appendix.csv")
        elif model == "A-GEM":
            results_path = os.path.join(results_dir, "results-agem-appendix.csv")
        else:
            results_path = ""
            # Throw an error for unknown model
            pass

        # Load the data
        df = pd.read_csv(results_path)


        # Convert the 'Mean_array' column to lists
        df['Mean_array'] = df['Mean_array'].apply(ast.literal_eval)

        # Append the value from the 'Mean' column to each list in 'Mean_array'
        df['Mean_array'] = df.apply(
            lambda row: row['Mean_array'] + [row['Mean']], axis=1
        )
        print(df[['Epochs','Mean_array']])

        # Print the updated DataFrame
        # Set legend labels
        legend_labels = ['task 1', 'task 2', 'task 3', 'task 4', 'task 5', 'mean']

        # Create the grouped bar chart
        x = np.arange(len(df['Epochs']))  # the label locations
        print(x)
        width = 0.15  # the width of the bars

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each task's data
        for i in range(len(legend_labels)):
            task_values = [row[i] + 0.001 for row in df['Mean_array']]
            # ax.bar(x + i * width, task_values, width, label=legend_labels[i])
            bars = ax.bar(x + i * width, task_values, width, label=legend_labels[i])
            print(task_values)

            # Add values on top of the bars
            for bar in bars:
                height = bar.get_height()
                print(height)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # x position
                    height,  # y position
                    f'{height:.2f}',  # text (formatted to 2 decimal places)
                    ha='center',  # horizontal alignment
                    va='bottom'  # vertical alignment
                )

        # Add labels, title, and legend
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Average accuracy [%]')
        # ax.set_title('Grouped Bar Chart of Mean Array per Epoch')
        ax.set_xticks(x + width * (len(legend_labels) - 1) / 2)
        ax.set_xticklabels(df['Epochs'])
        ax.legend(title="Metrics")

        # Set y-axis to log scale
        ax.set_yscale('log')

        # Show the plot
        plt.tight_layout()
        plt.show()



    # plt.tight_layout()
    # plt.show()

def generate_plots_forgetting():
    script_dir = Path(__file__).resolve().parent
    results_dir = os.path.join(script_dir, "Data/impact_of_hyperparameters/")
    results_path = os.path.join(results_dir, "results-lwf_mc-appendix.csv")
    df_lwf = pd.read_csv(results_path)

    results_path = os.path.join(results_dir, "results-agem-appendix.csv")
    df_agem = pd.read_csv(results_path)

    df = pd.DataFrame()

    df.insert(0, 'epochs', df_lwf['Epochs'])
    df.insert(1, 'lwf_f', df_lwf['Forgetting'])
    df.insert(2, 'agem_f', df_agem['Forgetting'])

    plt.plot(df['epochs'],df['lwf_f'],label="LwF-MC",marker='o')
    plt.plot(df['epochs'], df['agem_f'],label="A-GEM",marker='o')

    plt.xlabel("Epochs")
    plt.ylabel("Forgetting [%]")

    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    generate_plots_classes()
    generate_plots_forgetting()