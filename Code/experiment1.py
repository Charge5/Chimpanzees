import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import random
import os
import datetime
from src.activationregion.core import count_for_experiment

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

"""
Make sure to run this script from /Code directory.
This experiment aims at reproducing the results from Figure 3 of https://arxiv.org/abs/1906.00904.
Sine the method to count the number of activation regions was only outlined in the above paper,
we consider this experiment as a validation tool for our implementation.
We give special attention to the number of activation regions at initizialization as Theorem 5 gives
a probablistic upper bound.
"""

# Utils to log the experiment
def create_dir():
    current_time = datetime.datetime.now().isoformat()
    current_time = current_time.replace(":", "_").split(".")[0]
    results_output_path = r'replication_paper/'
    results_output_path = os.path.join(results_output_path, current_time)
    os.mkdir(results_output_path)
    return results_output_path

def save_params(params, save_path):
    file = os.path.join(save_path, r'parameters.txt')
    with open(file, "w") as file:
        for key, value in params.items():
            file.write(f'{key}: {value}\n')

# MLP model
class MLP(nn.Module):
    def __init__(self, n_in, width=16, depth=4):
        super(MLP, self).__init__()
        self.n_in = n_in
        layers = []
        layers.append(nn.Linear(n_in, width, bias=True))
        layers.append(nn.ReLU())
        for i in range(1, depth):
            layers.append(nn.Linear(width, width, bias=True))
            layers.append(nn.ReLU())
        self.classifier = nn.Linear(width, 10, bias=True)
        self._features = nn.Sequential(*layers)
        self.init_weights()
        self.n_neurons = self.count_neurons()

    def count_neurons(self):
        total_neurons = 0
        for layer in self._features:
            if isinstance(layer, nn.Linear):
                total_neurons += layer.out_features
        return total_neurons

    def init_weights(self):
        for layer in self._features:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(layer.bias, std=10**-6)
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.classifier.bias, std=10**-6)

    def forward(self, x):
        x = x.view(-1, self.n_in)
        for layer in self._features:
            x = layer(x)
        output = self.classifier(x)
        return output

# Load MNIST dataset
image_size = 16
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size,image_size)),
    transforms.Lambda(lambda x: x.view(image_size * image_size, 1)),
])
root_folder = 'src/mammoth/data/MNIST'
train_dataset = datasets.MNIST(root=root_folder, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=root_folder, train=False, download=True, transform=transform)

batch_size = 128
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Main function for the experiment:
# Train the model and count the regions at epochs given in count_vals
def train_and_count(model, lr, train_loader, n_epochs,
                    count_vals, init_vertices, with_closeup=True,
                    image_size=16, verbose=False):
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_step = len(train_loader)
    if with_closeup:
        closeup = np.array([0.02, 0.04, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5])
        count_vals = np.concatenate((np.array([0]), closeup, count_vals))
    else:
        count_vals = np.concatenate((np.array([0]), count_vals))

    region_evol = np.zeros((len(count_vals),))
    counts = count_for_experiment(model._features, init_vertices=init_vertices,
                                  MNIST_PATH=root_folder, image_size=image_size,
                                  n_planes=5)
    region_evol[0] = counts
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch==0 and with_closeup:
                if i in np.array(total_step * closeup, dtype=int):
                    if verbose:
                        print(f"Counting ...")
                    idx = np.where(np.array(total_step * closeup, dtype=int) == i)[0][0]
                    counts = count_for_experiment(model._features, init_vertices=init_vertices, 
                                  MNIST_PATH=root_folder, image_size=image_size,
                                  n_planes=5)
                    region_evol[idx+1] = counts

            if verbose and (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")

        if epoch + 1 in count_vals:
            if verbose:
                print("Counting ...")
                counts = count_for_experiment(model._features, init_vertices=init_vertices, 
                                MNIST_PATH=root_folder, image_size=image_size,
                                n_planes=5)
            idx = np.where(count_vals == epoch + 1)[0][0]
            region_evol[idx] = counts
    return count_vals, region_evol

# Generate and save the plot
def save_plot(counts_vals, curves, width, depth, save_path=r'replication_paper'):
    curves_arr = np.array(curves)/((model.n_neurons)**2)
    mean_curve = np.mean(curves_arr, axis=0)
    std_dev = np.std(curves_arr, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    # closeup
    axes[0].plot(counts_vals[:9], mean_curve[:9], label = f"depth:{depth}, width:{width}", color="blue", linewidth=2, marker='o')
    axes[0].fill_between(counts_vals[:9], mean_curve[:9] - std_dev[:9], mean_curve[:9] + std_dev[:9], color="lightblue", alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Number of Regions over \n squared number of neurons")
    axes[0].legend()
    axes[0].grid(True)

    # Full
    axes[1].plot(counts_vals, mean_curve, label = f"depth:{depth}, width:{width}", color="blue", linewidth=2, marker='o')
    axes[1].fill_between(counts_vals, mean_curve - std_dev, mean_curve + std_dev, color="lightblue", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Number of Regions over \n squared number of neurons")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    # save figure
    plt.savefig(os.path.join(save_path, r'experiment1.png'))

# Run the experiment :
dir_path = create_dir()

# Modify the parameters of the experiment directly in the dictionary
params = {
    'n_experiment': 5,
    'n_planes': 5,
    'width': 16,
    'depth': 4,
    'image_size': image_size,
    'lr': 0.001,
    'n_epochs': 10,
    'count_vals': np.array([1, 2, 3, 4, 5, 7, 8, 10]),
    'init_vertices': [[-500, -500], [-500, 500], [500, 500], [500, -500]]
}
save_params(params, dir_path)

curves = []
for i in range(params['n_experiment']):
    print(f"Experiment {i+1}/{params['n_experiment']}")
    model = MLP(n_in = params['image_size']**2, 
                width=params['width'], 
                depth=params['depth'])
    x_vals, region_evol = train_and_count(model, params['lr'], train_loader, params['n_epochs'],
                                          params['count_vals'], init_vertices=params['init_vertices'],
                                          with_closeup=True, image_size=params['image_size'], verbose=True)
    curves.append(region_evol)

curves = np.array(curves)
np.save(os.path.join(dir_path, r'curves'), curves)

save_plot(x_vals, curves, width=params['width'], depth=params['depth'], save_path=dir_path)