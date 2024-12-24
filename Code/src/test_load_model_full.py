import torch
import sys
sys.path.append('/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth')
import models


PATH  = r'/Users/thomaszilliox/Documents/git_repos/Chimpanzees/Code/src/mammoth/data/results/ETH/2024-12-24T10_05_07/model_task_1.pt'
model = torch.load(PATH, weights_only=False)
model.eval()
print(model)