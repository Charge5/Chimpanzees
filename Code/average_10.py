import  pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import  numpy as np


results_path = "/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/results10-merged.csv"
results_path = "/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/results20-merged.csv"


df = pd.read_csv(results_path)
df = df.drop(columns=['Date'])
# df = df.query("Seed != 10")
# print(df)

df = df.drop_duplicates()
# print(df)

depths = df['Depth'].unique()
epochs = df['Epochs'].unique()
widths = df['Width'].unique()

# d = depths[0]
# e = epochs[0]
# w = widths[0]

# Generate Cartesian product
combinations = list(product(depths, epochs, widths))
total_combinations = len(combinations)

mean_2 = []
std_2 = []
f_mean_2 = []
f_std_2 = []
mean_5 = []
std_5 = []
f_mean_5 = []
f_std_5 = []


for i, (d, e, w) in enumerate(combinations, start=1):
    data = df[(df['Depth']==d) & (df['Epochs']==e) & (df['Width']==w)]
    print(f"d: {d}, e: {e} and w: {w} -> mean = {data['Mean'].mean()} and std = {data['Mean'].std()}")
    if d == 2:
        mean_2.append(data['Mean'].mean())
        std_2.append(data['Mean'].std())
        f_mean_2.append(data['Forgetting'].mean())
        f_std_2.append(data['Forgetting'].std())

    else:
        mean_5.append(data['Mean'].mean())
        std_5.append(data['Mean'].std())
        f_mean_5.append(data['Forgetting'].mean())
        f_std_5.append(data['Forgetting'].std())

    if len(data['Seed']) != 10:
        print(f"d: {d}, e: {e} and w: {w}")
        for i,j in zip(data['Mean'],data['Seed']):
            print(f"mean: {i} and seed {j}")
        # print(f"mean: {data['Mean']} and seed {data['Seed']}")

mean_2 = np.array(mean_2)
std_2 = np.array(std_2)
mean_5 = np.array(mean_5)
std_5 = np.array(std_5)
f_mean_2 = np.array(f_mean_2)
f_std_2 = np.array(f_std_2)
f_mean_5 = np.array(f_mean_5)
f_std_5 = np.array(f_std_5)

# plot it!
t = widths
# fig, ax = plt.subplots(1)
# Create a second y-axis
# ax2 = ax.twinx()

# Create a figure with two subplots side-by-side


fig, axes = plt.subplots(2, 2, figsize=(12, 5))  # 1 row, 2 columns
for i,model in enumerate(["LwF-MC","A-GEM"]):

    # fig.suptitle(val, fontsize=12)


    model_1 = "depth: 2"
    model_2 = "depth: 5"

    axes[i][0].plot(t, mean_2, lw=2, label=model_1, color='blue')
    axes[i][0].plot(t, mean_5, lw=2, label=model_2, color='red')
    axes[i][0].fill_between(t, mean_2+std_2, mean_2-std_2, facecolor='blue', alpha=0.3)
    axes[i][0].fill_between(t, mean_5+std_5, mean_5-std_5, facecolor='red', alpha=0.3)

    # axes[0].set_title(r'Study of forgetting')
    axes[i][0].legend(loc='upper left')
    axes[i][0].set_xlabel('Width')
    axes[i][0].set_ylabel('$A_{T}$ [%]')
    axes[i][0].grid()

    axes[i][0].set_title(f"{model} average accuracy")

    axes[i][1].plot(t, f_mean_2, lw=2, label=model_1, color='blue')
    axes[i][1].plot(t, f_mean_5, lw=2, label=model_2, color='red')
    axes[i][1].fill_between(t, f_mean_2+f_std_2, f_mean_2-f_std_2, facecolor='blue', alpha=0.3)
    axes[i][1].fill_between(t, f_mean_5+f_std_5, f_mean_5-f_std_5, facecolor='red', alpha=0.3)

    # axes[1].set_title(r'Study of forgetting')
    axes[i][1].legend(loc='upper left')
    axes[i][1].set_xlabel('Width')
    axes[i][1].set_ylabel('$F_{T}$ [%]')
    axes[i][1].grid()

    axes[i][1].set_title(f"{model} forgetting")



plt.tight_layout()

plt.show()