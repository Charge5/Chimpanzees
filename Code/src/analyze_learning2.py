import os
import pandas as pd
import matplotlib.pyplot as plt


path =  '/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data/results/ETH/results-merge.csv'

# f = open(path,'r')
# data = f.read()
# print(type(data))

df = pd.read_csv(path)
# print(df)
for i in [2,5,10]:
    filtered_df = df[df['Depth'] == i]
    df_lr_e_10_001 = filtered_df[(filtered_df['Learning Rate'] == 0.01) & (filtered_df['Epochs'] == 10)]
    df_lr_e_10_0001 = filtered_df[(filtered_df['Learning Rate'] == 0.001) & (filtered_df['Epochs'] == 10)]
    df_lr_e_20_001 = filtered_df[(filtered_df['Learning Rate'] == 0.01) & (filtered_df['Epochs'] == 20)]
    df_lr_e_20_0001 = filtered_df[(filtered_df['Learning Rate'] == 0.001) & (filtered_df['Epochs'] == 20)]
    print(filtered_df)

    # Plot two specific columns
    plt.plot(df_lr_e_10_001['Width'], df_lr_e_10_001['Mean'], marker='o', color = 'b', label=f'LR = 0.01 - epochs 10')
    plt.plot(df_lr_e_20_001['Width'], df_lr_e_20_001['Mean'], marker='*', color = 'b', label=f'LR = 0.01 - epochs 20')

    plt.plot(df_lr_e_10_0001['Width'], df_lr_e_10_0001['Mean'], marker='o', color = 'r', label=f'LR = 0.001 - epochs 10')
    plt.plot(df_lr_e_20_0001['Width'], df_lr_e_20_0001['Mean'], marker='*', color = 'r', label=f'LR = 0.001 - epochs 20')

    # Add labels and title
    plt.xlabel('Width')
    plt.ylabel('Average Mean')
    plt.title(f'Effect of the width on the average mean with depth = {i}')
    plt.legend()

    # Show plot
    plt.show()