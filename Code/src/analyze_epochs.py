import os
import pandas as pd
import matplotlib.pyplot as plt


path =  '/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data/results/ETH/results4.csv'

# f = open(path,'r')
# data = f.read()
# print(type(data))

df = pd.read_csv(path)
print(df)
df_lr_e_5_001 = df[(df['Learning Rate'] == 0.001) & (df['Width'] == 5)]
df_lr_e_5_01 = df[(df['Learning Rate'] == 0.01) & (df['Width'] == 5)]
df_lr_e_15_001 = df[(df['Learning Rate'] == 0.001) & (df['Width'] == 15)]
df_lr_e_15_01 = df[(df['Learning Rate'] == 0.01) & (df['Width'] == 15)]
print(df_lr_e_15_01)
# Plot two specific columns
plt.plot(df_lr_e_5_01['Epochs'], df_lr_e_5_01['Mean'], marker='o', color = 'b', label=f'LR = 0.01, W=5')
plt.plot(df_lr_e_5_001['Epochs'], df_lr_e_5_001['Mean'], marker='*', color = 'b', label=f'LR = 0.001, W=5')
plt.plot(df_lr_e_15_01['Epochs'], df_lr_e_15_01['Mean'], marker='o', color = 'r', label=f'LR = 0.01, W=15')
plt.plot(df_lr_e_15_001['Epochs'], df_lr_e_15_001['Mean'], marker='*', color = 'r', label=f'LR = 0.001, W=15')



# Add labels and title
plt.xlabel('Number of epochs')
plt.ylabel('Average Mean')
plt.title(f'Depth = 2')
plt.legend()

# Show plot
plt.show()