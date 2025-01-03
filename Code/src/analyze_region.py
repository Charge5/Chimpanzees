import os
import pandas as pd
import matplotlib.pyplot as plt
import json


path =  '/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data/results/ETH/results4.csv'
region_path = '/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data.json'
# f = open(path,'r')
# data = f.read()
# print(type(data))

f = open(region_path,'r')
region_json = f.read()

region_data = json.loads(region_json)

for a in region_data:
    a['mlphiddensize'] = float(a['mlphiddensize'])
    a['lr'] = float(a['lr'])
    a['nepochs'] = float(a['nepochs'])
    try:
        a['mlphiddendepth'] = float(a['mlphiddendepth'])
    except:
        pass


df_region = pd.DataFrame(region_data)
# print(df_region)

df = pd.read_csv(path)

# print(df)
df_lr_e_5_001 = df[(df['Learning Rate'] == 0.001) & (df['Width'] == 5)]
df_lr_e_5_01 = df[(df['Learning Rate'] == 0.01) & (df['Width'] == 5)]
df_lr_e_15_001 = df[(df['Learning Rate'] == 0.001) & (df['Width'] == 15)]
# df_lr_e_15_01 = df[(df['Learning Rate'] == 0.01) & (df['Width'] == 15)]

# print(df_lr_e_15_001)
for i in [2,5,10]:
    df_1 = df_region[(df_region['lr'] == 0.001) & (df_region['mlphiddendepth'] == i)& (df_region['nepochs'] == 10)]
    df_no_duplicates = df_1[~df_1['mlphiddensize'].duplicated(keep='first')]
    sorted_df = df_no_duplicates.sort_values(by='mlphiddensize')
    plt.plot(sorted_df['mlphiddensize'], sorted_df['regions'], marker='o', color = 'r', label=f'LR = 0.001, epochs = 10')

    df_2 = df_region[(df_region['lr'] == 0.001) & (df_region['mlphiddendepth'] == i)& (df_region['nepochs'] == 20)]
    df_no_duplicates = df_2[~df_2['mlphiddensize'].duplicated(keep='first')]
    sorted_df = df_no_duplicates.sort_values(by='mlphiddensize')
    plt.plot(sorted_df['mlphiddensize'], sorted_df['regions'], marker='*', color = 'r', label=f'LR = 0.001, epochs = 20')

    df_3 = df_region[(df_region['lr'] == 0.01) & (df_region['mlphiddendepth'] == i)& (df_region['nepochs'] == 10)]
    df_no_duplicates = df_3[~df_3['mlphiddensize'].duplicated(keep='first')]
    sorted_df = df_no_duplicates.sort_values(by='mlphiddensize')
    plt.plot(sorted_df['mlphiddensize'], sorted_df['regions'], marker='o', color = 'b', label=f'LR = 0.01, epochs = 10')

    df_4 = df_region[(df_region['lr'] == 0.01) & (df_region['mlphiddendepth'] == i)& (df_region['nepochs'] == 20)]
    df_no_duplicates = df_4[~df_4['mlphiddensize'].duplicated(keep='first')]
    sorted_df = df_no_duplicates.sort_values(by='mlphiddensize')
    plt.plot(sorted_df['mlphiddensize'], sorted_df['regions'], marker='*', color = 'b', label=f'LR = 0.01, epochs = 20')



    # Add labels and title
    plt.xlabel('Number of epochs')
    plt.ylabel('Number of regions')
    plt.title(f'Effect of the width on the regions with depth = {i}')
    plt.legend()

    # Show plot
    plt.show()


