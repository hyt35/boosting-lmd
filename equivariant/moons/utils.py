import torch
import numpy as np
import os
import torch.nn as nn
from sklearn import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


class moons:
    def __init__(self, noise=0.05, dataset_path = 'dataset/moons', batch_size = 50):
        self.dataset_path = dataset_path
        self.noise = noise
        self.trainmoons = None
        self.batch_size = batch_size
        self.n_samples = None
    def trainset(self, n_samples = 500):
        if self.trainmoons is not None:
            return self.trainmoons
        create_paths("dataset", "moons")
        if not os.path.isfile(os.path.join(self.dataset_path, str(n_samples)+str(self.noise))):
            # create
            noisy_moons = datasets.make_moons(n_samples = n_samples, noise = self.noise)
            dat = torch.as_tensor(noisy_moons[0], dtype=torch.float32)
            label = torch.as_tensor(noisy_moons[1], dtype=torch.float32)
            label.unsqueeze_(1)
            torch.save([dat, label], os.path.join(self.dataset_path, str(n_samples)+str(self.noise)))
            self.n_samples = n_samples
            self.trainmoons = [dat, label]
            return self.trainmoons
        else:
            self.trainmoons = torch.load(os.path.join(self.dataset_path, str(n_samples)+str(self.noise)))
            self.n_samples = n_samples
            return self.trainmoons
    
    def train_minibatch(self):
        if self.trainmoons is None:
            raise Exception("train set not loaded")
        if len(self.trainmoons[1]) < self.batch_size:
            raise Exception("batch size larger than train set size")
        ind = np.random.choice(self.n_samples, self.batch_size, replace=False)

        with torch.no_grad():
            return [self.trainmoons[0][ind,:], self.trainmoons[1][ind]]
    

    def testset(self, n_samples = 50):
        noisy_moons = datasets.make_moons(n_samples = n_samples, noise = self.noise)
        dat = torch.as_tensor(noisy_moons[0], dtype=torch.float32)
        label = torch.as_tensor(noisy_moons[1], dtype=torch.float32)
        return dat, label

class Net(nn.Module):
    # Basic 1 layer neural network
    # should be sufficient for this.
    def __init__(self):
        super(Net, self).__init__()
        self.filters = nn.ModuleList([nn.Linear(2, 50),
                                      nn.Linear(50, 1)])
    def forward(self, x):
        out = self.filters[0](x)
        out = F.relu(out)
        out = self.filters[1](out)
        return out
    
class NetMNIST(nn.Module):
    def __init__(self):
        super(NetMNIST, self).__init__()
        self.filters = nn.ModuleList([nn.Linear(784, 50),
                                      nn.Linear(50, 10)])
    def forward(self, x):
        out = torch.flatten(x, start_dim=1)
        out = self.filters[0](out)
        out = F.relu(out)
        out = self.filters[1](out)

        # to be used with NLL loss
        # remove if using cross entropy
        out = F.log_softmax(out, dim=1)
        return out
def create_paths(*argv):
    path = None
    for arg in argv:
        if path is None:
            path = arg
        else:
            path = os.path.join(path, arg)
        if not os.path.exists(path):
            os.mkdir(path)   

    return path

# TODO Visualization tool
def vis_decision(model, fpath = None, title = None, hparams = None):
    # generates a boundary plot
    # since moons boundary is
    xs = np.linspace(-1.5,2.5,80)
    ys = np.linspace(-1.,2.,60)
    inp = torch.cartesian_prod(torch.as_tensor(xs, dtype=torch.float32), torch.as_tensor(ys, dtype=torch.float32))
    fig, ax = plt.subplots()

    pred = model(inp)<0.5
    pred = pred[:,0]
    ax.scatter(inp[pred,0], inp[pred,1], color="b", alpha=0.5)
    ax.scatter(inp[~pred,0], inp[~pred,1], color='r', alpha=0.5)

    if hparams is None:
        tempmoons = moons()
        point, labels = tempmoons.testset(300)
    else:
        moons_set = moons(noise = hparams['noise'], batch_size=hparams['train_batchsize'])
        point, labels = moons_set.trainset(n_samples = hparams['train_sample_count'])
        labels = labels[:,0]

    labels = labels==0
    
    ax.scatter(point[labels,0],point[labels,1], color="m", alpha=0.9)
    ax.scatter(point[~labels,0],point[~labels,1], color="c", alpha=0.9)

    if title is not None:
        fig.suptitle(title)
    if fpath is not None:
        fig.savefig(fpath)
    else:
        return fig,ax
