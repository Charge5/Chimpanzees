import  pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import  numpy as np




def get_metrics(path):
    df = pd.read_csv(results_path)
    df = df.drop(columns=['Date'])
    # df = df.query("Seed != 10")
    # print(df)

    df = df.drop_duplicates()
    # print(df)

    depths = df['Depth'].unique()
    epochs = df['Epochs'].unique()
    widths = df['Width'].unique()

    # Generate Cartesian product
    combinations = list(product(depths, epochs, widths))
    total_combinations = len(combinations)

    # Extract unique values
    depths = df['Depth'].unique()
    epochs = df['Epochs'].unique()
    widths = df['Width'].unique()

    # Generate Cartesian product
    combinations = product(depths, epochs, widths)

    # Initialize results dictionary
    results = {2: {'mean': [], 'std': [], 'f_mean': [], 'f_std': []},
               5: {'mean': [], 'std': [], 'f_mean': [], 'f_std': []}}

    # Loop through combinations
    for d, e, w in combinations:
        data = df[(df['Depth'] == d) & (df['Epochs'] == e) & (df['Width'] == w)]

        # Compute mean and std for current group
        mean_value = data['Mean'].mean()
        std_value = data['Mean'].std()
        f_mean_value = data['Forgetting'].mean()
        f_std_value = data['Forgetting'].std()

        print(f"d: {d}, e: {e}, w: {w} -> mean = {mean_value:.2f}, std = {std_value:.2f}")

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

    return results, widths



# plot it!
# t = widths
# fig, ax = plt.subplots(1)
# Create a second y-axis
# ax2 = ax.twinx()

# Create a figure with two subplots side-by-side


fig, axes = plt.subplots(2, 2, figsize=(12, 5))  # 1 row, 2 columns
for i,model in enumerate(["LwF-MC","A-GEM"]):

    if model == "LwF-MC":
        results_path = "/Code/Data/impact_of_hyperparameters/results-lwf_mc.csv"
    else:
        results_path = "/Code/Data/impact_of_hyperparameters/results-agem.csv"

    results,t = get_metrics(results_path)
    # fig.suptitle(val, fontsize=12)

    mean_2, std_2, f_mean_2, f_std_2 = results[2]['mean'], results[2]['std'], results[2]['f_mean'], results[2]['f_std']
    mean_5, std_5, f_mean_5, f_std_5 = results[5]['mean'], results[5]['std'], results[5]['f_mean'], results[5]['f_std']


    model_1 = "depth: 2"
    model_2 = "depth: 5"

    axes[i][0].plot(t, mean_2, lw=2, label=model_1, color='blue')
    axes[i][0].plot(t, mean_5, lw=2, label=model_2, color='red')
    axes[i][0].fill_between(t, mean_2+std_2, mean_2-std_2, facecolor='blue', alpha=0.3)
    axes[i][0].fill_between(t, mean_5+std_5, mean_5-std_5, facecolor='red', alpha=0.3)

    # axes[0].set_title(r'Study of forgetting')
    axes[i][0].legend()
    axes[i][0].set_xlabel('Width')
    axes[i][0].set_ylabel('$A_{T}$ [%]')
    axes[i][0].grid()

    axes[i][0].set_title(f"{model} average accuracy")

    axes[i][1].plot(t, f_mean_2, lw=2, label=model_1, color='blue')
    axes[i][1].plot(t, f_mean_5, lw=2, label=model_2, color='red')
    axes[i][1].fill_between(t, f_mean_2+f_std_2, f_mean_2-f_std_2, facecolor='blue', alpha=0.3)
    axes[i][1].fill_between(t, f_mean_5+f_std_5, f_mean_5-f_std_5, facecolor='red', alpha=0.3)

    # axes[1].set_title(r'Study of forgetting')
    # axes[i][1].legend(loc='upper left')
    axes[i][1].legend()
    axes[i][1].set_xlabel('Width')
    axes[i][1].set_ylabel('$F_{T}$ [%]')
    axes[i][1].grid()

    axes[i][1].set_title(f"{model} forgetting")



plt.tight_layout()

plt.show()