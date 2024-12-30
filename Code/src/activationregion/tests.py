import torch
import random
import numpy as np
import torch.nn as nn
import random
import os
import sys

ROOT_DIR = os.getcwd()
SRC_PATH = os.path.join(ROOT_DIR, r'src/activationregion')
sys.path.append(SRC_PATH)

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

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

image_size=16
model = MLP(n_in=image_size**2, width=32, depth=2)

from src.activationregion.utils.utils import random_mnist_images
MNIST_PATH = 'src/mammoth/data/MNIST'
images_plane = random_mnist_images(3, MNIST_PATH, image_size=image_size)

"""
1st method: Exact counting
"""
from src.activationregion.core import exact_count_2D
regions = exact_count_2D(model._features, images_plane, init_vertices=[[-1.5, -1.5], [-1.5, 0.5], [0.5, 0.5], [0.5, -1.5]])
print(len(regions))
#assert len(regions) == 5473

"""
2nd method: Sample counting
"""
from src.activationregion.core import sample_count_2D
n_regions = sample_count_2D(model._features, images_plane, domain_bounds=[-1.5, 0.5], n_samples=1000, return_inverse=False)
print(n_regions)
#assert n_regions == 5088