import os
import re
import ast

import numpy as np

import numpy as np

# Get the results directory
results_dir = "mammoth/data/results"
root_dir = os.getcwd()
results_dir = os.path.join(root_dir,results_dir)

# Go to the class-il as it's the one that can demonstrate the catastrophic forgetting
class_il_dir = "class-il"
dataset = "seq-mnist"
model = "lwf_mc" # Continous learning model
results_file = os.path.join(results_dir,class_il_dir,dataset,model,"logs.pyd")

# print(results_file)

f = open(results_file, "r")
results = f.readlines()
last_results = results[-1]
last_results = last_results.replace("np.float64(","").replace(")","").replace("device(type=","")

# Convert the string to a dictionary
last_results = ast.literal_eval(last_results)

for i in range(5):
    print(f"Results for task{i}")
    print(last_results[f'accmean_task{i+1}'])
    filtered_dict = {k: v for k, v in last_results.items() if re.match(rf'accuracy_\d+_task{i}', k)}
    print(filtered_dict)

print(last_results[f'accmean_task{5}'])