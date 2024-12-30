import torch
from torchvision import datasets, transforms
import random

def random_mnist_images(n, PATH=r'data/MNIST', image_size=28):
    """
    Randomly select n images from MNIST training set, located at PATH.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size,image_size)),
        transforms.Lambda(lambda x: x.view(image_size * image_size, 1)),
    ])
    train_dataset = datasets.MNIST(PATH,
                            train=True, download=True, transform=transform)
    
    idx = [random.randint(0, len(train_dataset)-1) for i in range(n)]
    images = []
    for i in range(len(idx)):
        images.append(train_dataset[idx[i]][0])
    images = torch.cat(images, dim=1).T
    images = images[..., None]
    return images

def images_for_experiment(n_planes, MNIST_PATH=r'data/MNIST', through_zero=False, image_size=28):
    image_set = []
    for i in range(n_planes):
        if through_zero:
            images_plane = random_mnist_images(2, MNIST_PATH, image_size=image_size)
            zero = torch.zeros((1, image_size**2, 1), dtype=torch.float32)
            images_plane = torch.cat((images_plane, zero), dim=0)
        else:
            images_plane = random_mnist_images(3, MNIST_PATH, image_size=image_size)
        image_set.append(images_plane)
    return image_set