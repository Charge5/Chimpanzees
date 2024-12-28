import os
import pandas as pd
import matplotlib.pyplot as plt


path =  '/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data/results/ETH/results4.csv'

# f = open(path,'r')
# data = f.read()
# print(type(data))

df = pd.read_csv(path)
print(df)


# Plot two specific columns
plt.plot(df['Epochs'], df['Mean'], marker='o', color = 'b', label=f'LR = 0.001')

# Add labels and title
plt.xlabel('Number of epochs')
plt.ylabel('Average Mean')
plt.title(f'Depth = 2 and Width = 15')
plt.legend()

# Show plot
plt.show()