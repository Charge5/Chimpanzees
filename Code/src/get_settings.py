import os
import re
import ast

import numpy as np

import numpy as np

# Get the results directory
results_dir = "mammoth/data/results/ETH"
root_dir = os.getcwd()
results_dir = os.path.join(root_dir,results_dir)

print(results_dir)
result_dir = os.path.join(results_dir,"2024-12-23T16_32_49")
print(result_dir)