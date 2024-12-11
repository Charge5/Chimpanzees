import numpy as np
import torch
import metrics
import matplotlib.pyplot as plt
from itertools import islice
import plotly.graph_objects as go



np.random.seed(0)


class FeedForwardNN2D(torch.nn.Module):
    def __init__(self):
        super(FeedForwardNN2D, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(2, 2))
        self.layers.append(torch.nn.Linear(2, 2))
        self.layers.append(torch.nn.Linear(2, 1))
        self.init_weights_random()

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.ones_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def init_weights_random(self):
        for layer in self.layers:
            torch.nn.init.normal_(layer.weight)
            torch.nn.init.normal_(layer.bias)

    def forward(self, x):
        activation = []
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
            activation.append((x > 0).squeeze().int())

        x = self.layers[-1](x)
        activation = torch.cat(activation, dim=1)
        return x, activation


model = FeedForwardNN2D()

n_samples = 800
boundary = 100
X1 = np.linspace(-boundary, boundary, n_samples)
X2 = np.linspace(-boundary, boundary, n_samples)
X1, X2 = np.meshgrid(X1, X2)
X = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)
X = torch.tensor(X).float().view(n_samples**2, 2)

with torch.no_grad():
    Y, activations = model(X)

unique_activations, inverse_indices = torch.unique(activations, dim=0, return_inverse=True)
print(f"Unique activations: {unique_activations.shape[0]}")

metrics.plot_activation(X1, X2, inverse_indices, n_samples,unique_activations)



