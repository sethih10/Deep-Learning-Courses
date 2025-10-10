import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import warnings, sys

import torch
import torchvision
from torchvision import transforms
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange


def select_data_dir(data_dir='../data'):
    data_dir = '/coursedata' if os.path.isdir('/coursedata') else data_dir
    print('The data directory is %s' % data_dir)
    return data_dir


def get_validation_mode():
    try:
        return bool(os.environ['NBGRADER_VALIDATING'])
    except:
        return False


def save_model(model, filename, confirm=True):
    if confirm:
        try:
            save = input('Do you want to save the model (type yes to confirm)? ').lower()
            if save != 'yes':
                print('Model not saved.')
                return
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')

    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    filesize = os.path.getsize(filename)
    if filesize > 30000000:
        raise 'The file size should be smaller than 30Mb. Please try to reduce the number of model parameters.'
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()


def show_images(images, ncol=12, figsize=(8,8), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    out = rearrange(images, '(b1 b2) c h w -> c (b1 h) (b2 w)', b2=ncol).cpu()
    if out.shape[0] == 1:
        ax.matshow(out[0], **kwargs)
    else:
        ax.imshow(out.permute((1, 2, 0)), **kwargs)
    display.display(fig)
    plt.close(fig)


def plot_generated_samples_(samples, ncol=12):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('off')
    ax.imshow(
        np.transpose(
            utils.make_grid(samples, nrow=ncol, padding=0, normalize=True).cpu(),
            (1,2,0)
        )
     )
    display.display(fig)
    plt.close(fig)

def customwarn(message, category, filename, lineno, file=None, line=None):
    sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))


class MNIST(torch.utils.data.Dataset):
    def __init__(self, data_dir, train = True, N = None):
        self.transform = transforms.Compose([
            transforms.ToTensor() # Transform to tensor
            #transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))
        ])
        if train:
            self.mnist = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=self.transform)
        else:
            self.mnist = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=self.transform)
        if N is not None:
            random_indeces = np.random.choice(len(self.mnist), N, replace=False)
            self.mnist.data = self.mnist.data[random_indeces]
            self.mnist.targets = self.mnist.targets[random_indeces]
    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        data = data/255
        return data, target, index

    def __len__(self):
        return len(self.data)