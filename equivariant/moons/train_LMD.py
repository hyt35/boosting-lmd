import torch
import utils, mirror_maps, generalization_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
np.random.seed(0)

device = 'cuda'

class DiagonalWeightedL2(nn.Module):
    def __init__(self, dims, n_iters = 50):
        super(DiagonalWeightedL2, self).__init__()
        self.param_weights = nn.Parameter(torch.ones(dims).to(device))
        # self.stepsize = nn.Parameter(0.05*torch.ones(n_iters).to(device))
        self.stepsize = 0.05*torch.ones(n_iters).to(device)
        
    def zero_clip_weights(self):
        with torch.no_grad():
            self.param_weights.clamp_(min = 0.001)
            self.stepsize.clamp_(min=0.001, max=0.1)
    def fwd_model(self, x):
        return x.mul(self.param_weights)

    def bwd_model(self, x):
        return x.mul(torch.reciprocal(self.param_weights))
    
    def forward(self, x, gradFun):
        with torch.no_grad():
            fwd = x
            for ss in self.stepsize:
                fwd = fwd - ss * torch.mul(self.param_weights, gradFun(fwd))
        return fwd

def train_model(hparams, experiment_id):
    utils.create_paths(experiment_id)
    
    moons = utils.moons(noise = hparams['noise'], batch_size=hparams['train_batchsize'])
    moons.trainset(n_samples = hparams['train_sample_count'])
    num_inits = hparams['num_inits']

    dat, label = moons.train_minibatch()
    dat = dat.to(device)
    label = label.to(device)

    lossFn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    dat.unsqueeze_(0)
    dat = dat.repeat(num_inits, 1, 1)
    label = label.repeat(1, num_inits)
    # Do training
    # print(model.filters[1].weight)
    # matplotlib stuff

    def nn_fwd(params):
        weight_1 = params[:, :100].reshape(-1, 2,50) #2 -> 10
        bias_1 = params[:,100:150] 
        weight_2 = params[:,150:200].reshape(-1, 50,1) # 10 -> 1
        bias_2 = params[:,200:]
        # evaluate
        # print(torch.matmul(dat, weight_1).shape)
        out = torch.matmul(dat, weight_1) + bias_1[:,None,:]
        out = F.relu(out)
        out = torch.matmul(out, weight_2) + bias_2[:,None,:]

        return out
    def nn_loss(params):
        out = nn_fwd(params)
        return lossFn(out[:,:,0], label.T) / hparams['train_sample_count'] # divide by moons batchsize
    
    def nn_loss_grad(layer_mat):
        return autograd.grad(nn_loss(layer_mat), layer_mat)[0]

    def create_param_lists(num_inits):
        weight_1 = (2*torch.rand(num_inits, 100) - 1)/(np.sqrt(2))
        bias_1 = (2*torch.rand(num_inits, 50) - 1)/(np.sqrt(2))
        weight_2 = (2*torch.rand(num_inits, 50) - 1)/(np.sqrt(50))
        bias_2 = (2*torch.rand(num_inits, 1) - 1)/(np.sqrt(50))
        catted = torch.cat([weight_1, bias_1, weight_2, bias_2], dim=1)
        catted = catted.to(device)
        return catted
    

    lmd_model = DiagonalWeightedL2(dims = 201, n_iters = hparams['lmd_maxitr'])
    # for p in lmd_model.parameters():
    #     print(p)
    lmd_model.train()
    opt = torch.optim.SGD(lmd_model.parameters(), lr = 1e-3)
    for epoch in range(hparams['epochs']+1):
        params = create_param_lists(num_inits)
        
        fwd = params
        fwd.requires_grad_()
        err = 0.
        for stepsize in lmd_model.stepsize:
            fwd_grad = nn_loss_grad(fwd)
            fwd = fwd - stepsize * torch.mul(fwd_grad, lmd_model.param_weights)
            fwd_loss = nn_loss(fwd)
            err += fwd_loss

        if epoch % 10 == 0:
            print(epoch, fwd_loss)

        opt.zero_grad()
        err.backward()
        opt.step()
        lmd_model.zero_clip_weights()
        if epoch % 100 == 0:
            print(lmd_model.param_weights)
    # check accuracy
        if epoch % 200 == 0:
            pred = torch.where(nn_fwd(fwd)>0., 1., 0.)
            test_acc = torch.sum(pred[:,:,0] == label.T)/(hparams['train_batchsize']*hparams['num_inits'])
            print(test_acc)
    # TODO checkpointing
    # TODO logging and displaying loss behavior
        if epoch % 1000 == 0 and epoch != 0:
            torch.save(lmd_model.state_dict(), "checkpoints/weightedl2/"+str(epoch))
        


if __name__ == '__main__':
    #################
    hparams = dict(epochs = 25000,
                lmd_maxitr = 100, 
                num_inits = 100,
                warmstart = False,
                noise = 0.05,
                train_sample_count = 500,
                train_batchsize = 500,
                lr = 5e-2
                )
    ###########

    torch.manual_seed(0)
    np.random.seed(0)
    model = utils.Net()
    model.train()

    train_model(hparams, experiment_id = 'test')
    utils.vis_decision(model, "figs/weighted_l2", title = r"weighted $L_2$", hparams = hparams)

