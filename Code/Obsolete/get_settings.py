import os
import re
import ast

import numpy as np

import numpy as np

# Get the results directory
results_dir = "mammoth/data/results/ETH"
root_dir = os.getcwd()
results_dir = os.path.join(root_dir,results_dir)

# print(results_dir)
result_dir = os.path.join(results_dir,"2024-12-27T01_08_45")
settings_dir = os.path.join(result_dir,"training_settings.txt")

# print(result_dir)
datas = []
for dir in os.listdir(results_dir):
    # print(dir)
    result_dir = os.path.join(results_dir, dir)
    if os.path.isdir(result_dir):
        # print(result_dir)
        settings_dir = os.path.join(result_dir, "training_settings.txt")
        model_path = os.path.join(result_dir, "model_task_5.pt")
        if os.path.exists(model_path):
            f = open(settings_dir)
            input_string = f.readline()
            # print(input_string)

            data = {}
            for text in input_string.split("--"):
                # print(text)
                text = text.replace("_","")
                # Regex pattern to split text from numbers
                pattern = r"([a-zA-Z]+)(\d+(\.\d+)?)"

                # Find matches
                match = re.match(pattern, text)

                if match:
                    text = match.group(1)
                    numbers = match.group(2)
                    # print(f"Text: {text}, Numbers: {numbers}")
                    data[text] = numbers
            data["path"] =dir
            datas.append(data)
# print(datas)
data_clean = []
for i in datas:
    try:
        i["mlphiddensize"]
        data_clean.append(i)
    except:
        pass
# print(data_clean)

import json
data_str = json.dumps(data_clean)
f = open("settings.json","w")
f.write(data_str)
f.close()

