# Train the spline-based LMD for CNNsmall architecture on omniglot task

import torch
import torchvision
import torchvision.transforms as transforms
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import time
import tqdm
import os
import numpy as np
import torch.autograd as autograd
import logging
from datetime import datetime
device = 'cuda'

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class CNNModelSmall(nn.Module):
    # 3x3 convolution
    def __init__(self):
        super(CNNModelSmall, self).__init__()
        
        self.filters = nn.ModuleList([nn.MaxPool2d(2,2),
                                      nn.Conv2d(1,8,3), #(14,14) -> (8,12,12)
                                      nn.ReLU(),
                                      nn.MaxPool2d(2,2), # (32, 26, 26) -> (32, 6,6)
                                      nn.Flatten(),
                                      nn.Linear(288, 10)
                                      ])

    def forward(self, x):
        for filter in self.filters:
            x = filter(x)
        out = F.log_softmax(x, dim=1)
        return out

class FiveLayerDense(nn.Module):
    def __init__(self):
        super(FiveLayerDense, self).__init__()
        self.filters = nn.ModuleList([nn.Flatten(),
                                      nn.Linear(784,50), nn.ReLU(),
                                      nn.Linear(50,40), nn.ReLU(),
                                      nn.Linear(40,30), nn.ReLU(),
                                      nn.Linear(30,20), nn.ReLU(),
                                      nn.Linear(20,10)])
    def forward(self, x):
        for filter in self.filters:
            x = filter(x)
        out = F.log_softmax(x, dim=1)
        return out
    
def test_acc(pred, target):
    return torch.sum(torch.argmax(pred, dim=2) == target) / (len(target) * pred.shape[0])

def CNNToParams(model):
    params = torch.cat([torch.flatten(filter) for filter in model.parameters()])
    size_array = []
    for p in model.parameters():
        size_array.append(p.shape)

    return params, size_array


def create_params(size_array, num_inits, mode=["conv", "conv", "dense"], device = 'cuda'):
    # returns vector of shape [num_inits, n_params] where n_params is total number
    # of parameters in the network
    i=0
    params_flat = torch.empty([num_inits, 0]).to(device)
    for m in mode:
        weight = torch.rand((num_inits,) + size_array[i]).to(device) # kernel weight
        bias = torch.rand((num_inits,) + size_array[i+1]).to(device)
        if m == "conv":
            conv_size = size_array[i] # input channels

            n_in_channels = conv_size[1]
            kernel_size = np.prod(conv_size[2:])
            
            # fix intializations
            # https://pytorch.org/docs/1.12/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
            scale = np.sqrt(1/(n_in_channels * kernel_size)) # sqrt{k} 

            weight = (2 * weight - 1) * scale # scale to Unif[-sqrt{k}, sqrt{k}]
            bias = (2 * bias - 1) * scale  # same
            i = i+2
        elif m == "dense":
            # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
            dense_size = size_array[i]
            n_in_channels = dense_size[1]
            scale = np.sqrt(1/n_in_channels)

            weight = (2 * weight - 1) * scale
            bias = (2 * bias - 1) * scale
            i = i+2
        params_flat = torch.cat((params_flat, weight.view(num_inits, -1), bias.view(num_inits, -1)), dim=1)
        # concatenate
    return params_flat # [num_inits, n_params] with standard Pytorch initializations


class ParamEvaler(nn.Module):
    # to easily evaluate the flattened parameters
    def __init__(self, num_inits, size_array, model_instance ="CNNModel", model_mode=["conv", "dense"]):
        super().__init__()
        self.device = device
        self.lossFn = torch.nn.NLLLoss(reduction='mean')
        self.num_inits = num_inits
        self.model_mode = model_mode
        self.size_array = size_array
        self.model_instance = model_instance


        flat_sizes = list(map(lambda x: np.prod(x), size_array))
        self.breakpoints = np.concatenate(([0],np.cumsum(flat_sizes)))

    def nn_fwd(self,params, batch):
        # print(batch.shape)
        if self.model_instance == "CNNModelSmall":
            foo = nn.MaxPool2d(2,2)
            batch = foo(batch)
        batch = torch.unsqueeze(batch, 0)
        batch = batch.to(self.device)
        batch.requires_grad_(False)
        # label = label.to(self.device)
        # label.requires_grad_(False)

        curr_batch = batch.repeat(self.num_inits, 1,1,1,1)
        out = curr_batch
        out.requires_grad_(False)

        i=0
        if self.model_instance == "CNNModel5":
            pool_layer = nn.MaxPool2d(3,2)
        elif self.model_instance == "CNNModel" or self.model_instance == "CNNModelSmall":
            pool_layer = nn.MaxPool2d(2,2)
            pool_layer_3d = nn.MaxPool3d((1,2,2), (1,2,2))
        else:
            raise Exception("Model not implemented")

        
        for mode in self.model_mode:
            weight = params[:, self.breakpoints[i]:self.breakpoints[i+1]]
            bias = params[:, self.breakpoints[i+1]:self.breakpoints[i+2]]
            weight = weight.view((self.num_inits,) + self.size_array[i])
            bias = bias.view((self.num_inits,) + self.size_array[i+1])

            if mode == "conv":
                curr_shape = out.shape
                N = curr_shape[1]
                C_in = weight.shape[2]
                C_out = weight.shape[1]

                w_shape = weight.shape

                weight = weight.reshape(self.num_inits * C_out, *w_shape[-3:])
                bias = bias.flatten()

                out_fast = out.reshape(self.num_inits, N, C_in, *curr_shape[-2:]).transpose(0,1).reshape(N, self.num_inits * C_in, *curr_shape[3:])
                out_grouped = F.conv2d(out_fast, weight, bias, groups=self.num_inits)

                out_grouped_shape = out_grouped.shape

                out = out_grouped.reshape(N, self.num_inits,C_out, *out_grouped_shape[-2:]).transpose(0,1)

                out = F.relu(out)

                out = pool_layer_3d(out)


            elif mode == "dense":
                if len(out.shape) == 5:
                    # flattening image, from [num_inits, N, C, H, W] -> [num_inits, N, C_new]
                    out = torch.flatten(out, start_dim=-3) 
                elif len(out.shape) != 3:
                    raise Exception("Unexpected shape")
                out = torch.matmul(out, torch.transpose(weight,-1,-2)) + bias[:,None,:] # 
                
            i = i+2

        out = F.log_softmax(out, dim=2)
        # print('out', out.shape)
        return out #shape (n_inits, batchsize, 10), hopefully


    def nn_loss(self,params, batch, label):
        curr_label = label.repeat(self.num_inits)
        out = self.nn_fwd(params, batch)
        return self.lossFn(out.view(-1,10), curr_label) * self.num_inits #/ hparams['train_sample_count'] # divide by moons batchsize
    
    def nn_loss_grad(self,layer_mat, batch, label):
        layer_mat_dupe = layer_mat.requires_grad_()
        return autograd.grad(self.nn_loss(layer_mat_dupe, batch, label), layer_mat_dupe)[0]
    

class AdHocWeightedL2(nn.Module):
    # Just holds parameters. 
    def __init__(self, groups, size_array, num_inits, n_iters = 50):
        # n_groups: list of list-like
        super(AdHocWeightedL2, self).__init__()
        self.groups = groups
        # Create parameters according to groups
        self.group_params = nn.ParameterList([nn.Parameter(torch.ones(shape).to(device)) for shape in groups])
        self.size_array = size_array
        # self.stepsize = nn.Parameter(0.05*torch.ones(n_iters).to(device))
        self.stepsize = 0.01*torch.ones(n_iters).to(device)
        self.num_inits = num_inits # number of nn inits - for convenience

        # checks
        if len(self.size_array) != len(self.groups):
            raise Exception("number of lmd groups and parameter groups is different")
        
    def zero_clip_weights(self):
        with torch.no_grad():
            for p in self.group_params:
                p.clamp_(min = 0.001)
            self.stepsize.clamp_(min=0.001, max=0.1)

    def fwd_model(self, x):
        if len(x) != len(self.groups):
            raise Exception("length of passed gradient vectors must be same as lmd")
        
        outs = []
        for x_, param, size in zip(x, self.group_params, self.size_array):
            # print(x_.shape, param.shape)
            out = torch.mul(x_.view((-1,)+size), torch.reciprocal(param)).view(self.num_inits,-1)
            outs.append(out)
            
        
        return outs

    def bwd_model(self, x):
        if len(x) != len(self.groups):
            raise Exception("length of passed gradient vectors must be same as lmd")
        
        outs = []
        for x_, param, size in zip(x, self.group_params, self.size_array):
            out = torch.mul(x_.view((-1,)+size), param).view(self.num_inits,-1)
            outs.append(out)
        
        return outs
    
    def transform_gradients(self, x):
        outs = []
        
        for x_, param, size in zip(x, self.group_params, self.size_array):
            out = torch.mul(x_.view(-1,*size), param).flatten(start_dim=1)
            outs.append(out)
        
        return outs

    def forward(self, x, gradFun):
        with torch.no_grad():
            fwd = x
            for ss in self.stepsize:
                fwd = fwd - ss * torch.mul(self.param_weights, gradFun(fwd))
        return fwd
    

lmd_hparams = dict(ckpt_path = "/local/scratch/public/hyt35/md-nn-generalization/omniglot_experiments/ckpt/CNNsmall",
                model_ver = "CNNModelSmall",
                groups=[[3,3],[1], # Conv 3x3
                        [10,1],[10] # Dense 800 -> 10. Weight matrix of shape [10,800]
                        ],
                model_mode=["conv","dense"],
                # lmd_maxitr = 120, 
                num_inits = 50,
                num_optims = 100,
                meta_epochs = 500000,
                valid_freq = 200,
                valid_num_tests = 10,
                checkpoint_freq = 5000,
                meta_batchsize = 1, # number of tasks/new network initialization sets before doing a LMD model update
                  # number of inner optimization loops
                ) 

_, size_array = CNNToParams(CNNModelSmall())
lmd_model = AdHocWeightedL2(groups=lmd_hparams['groups'], 
                            size_array=size_array, num_inits=lmd_hparams["num_inits"], n_iters = lmd_hparams['num_optims'])
# load pretrained
# lmd_model.load_state_dict(torch.load("/local/scratch/public/hyt35/md-nn-generalization/checkpoints/mnistfull_cnnsmall_partequiv/20000"))
lmd_model.train()

opt = torch.optim.SGD(lmd_model.parameters(), lr = 5e-3)


param_evaler = ParamEvaler(lmd_hparams["num_inits"], size_array, model_instance = lmd_hparams["model_ver"],model_mode = lmd_hparams["model_mode"])
breakpoints = param_evaler.breakpoints

dataset = omniglot("data", ways=10, shots=5, test_shots=15, meta_train=True, download=True)

dataset_valid = omniglot("data", ways=10, shots=5, test_shots=15, meta_test=True, download=True)
# dataloader_valid = BatchMetaDataLoader(dataset_valid, batch_size=lmd_hparams["meta_batchsize"], num_workers=4)# print(len(dataloader))

# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)# print(len(dataloader))
# ctr = 2

lossFn = torch.nn.NLLLoss(reduction='mean')

# opt = 'LAMD'
# opt = 'LMD'



dataloader = BatchMetaDataLoader(dataset, batch_size=lmd_hparams["meta_batchsize"], num_workers=4)# print(len(dataloader))
dataloader_valid = BatchMetaDataLoader(dataset_valid, batch_size=lmd_hparams["meta_batchsize"], num_workers=1)# print(len(dataloader))

logger = setup_logger('logger', "logs/log.log")
now = datetime.now()
logger.info("Start train " + now.strftime("%d-%m %H:%M"))

# avg_start, avg_end = 0,0
# avgs = torch.zeros(num_optims).to(device)
# start = time.time()
for meta_epoch, batch in tqdm.tqdm(zip(range(lmd_hparams["meta_epochs"]), dataloader), total=lmd_hparams['meta_epochs']):
    train_inputs, train_targets = batch["train"]
    test_inputs, test_targets = batch["test"]

    train_inputs, train_targets, test_inputs, test_targets = train_inputs.to(device), train_targets.to(device), test_inputs.to(device), test_targets.to(device)

    err = 0.

    opt.zero_grad()
    for task_idx, (train_input, train_target, test_input,
        test_target) in enumerate(zip(train_inputs, train_targets,
        test_inputs, test_targets)):

        fwd = create_params(size_array, lmd_hparams['num_inits'], mode=lmd_hparams["model_mode"]).to(device)

        param_decomp = [fwd[:, breakpoints[ctr]:breakpoints[ctr+1]] for ctr in range(len(breakpoints)-1)]
        params_dual = lmd_model.fwd_model(param_decomp)

        for i in range(lmd_hparams["num_optims"]):
            params_primal = [fwd[:, breakpoints[i]:breakpoints[i+1]] for i in range(len(breakpoints)-1)]
            fwd_grad = param_evaler.nn_loss_grad(fwd, train_input, train_target)
            # Split
            fwd_grad_list = [fwd_grad[:, breakpoints[i]:breakpoints[i+1]] for i in range(len(breakpoints)-1)]
            transformed_gradients = lmd_model.transform_gradients(fwd_grad_list)
            for ctr, t_grad in enumerate(transformed_gradients):
                params_primal[ctr] = params_primal[ctr] - lmd_model.stepsize[i] * t_grad
            
            fwd = torch.cat(params_primal, dim=1)

            fwd_loss = param_evaler.nn_loss(fwd, train_input, train_target)
            # fwd_loss.backward()
            err += fwd_loss
        err.backward(retain_graph = True)


    opt.step()
    lmd_model.zero_clip_weights()


    # TODO logging and checkpointing

    ## Validation
    if meta_epoch % lmd_hparams["valid_freq"] == 0:
        avg_start, avg_end = 0., 0.
        avg_loss = 0.
        for _ in range(lmd_hparams["valid_num_tests"]):
            batch = next(iter(dataloader_valid))
            train_inputs, train_targets = batch["train"]
            test_inputs, test_targets = batch["test"]
            train_inputs, train_targets, test_inputs, test_targets = train_inputs.to(device), train_targets.to(device), test_inputs.to(device), test_targets.to(device)

            lmd_model.eval()
            


            for task_idx, (train_input, train_target, test_input,
                test_target) in enumerate(zip(train_inputs, train_targets,
                test_inputs, test_targets)):

                fwd = create_params(size_array, lmd_hparams['num_inits'], mode=lmd_hparams["model_mode"]).to(device)

                param_decomp = [fwd[:, breakpoints[ctr]:breakpoints[ctr+1]] for ctr in range(len(breakpoints)-1)]
                params_dual = lmd_model.fwd_model(param_decomp)
                avg_start += test_acc(param_evaler.nn_fwd(fwd,test_input), test_target).item()
                for i in range(lmd_hparams["num_optims"]):
                    params_primal = [fwd[:, breakpoints[i]:breakpoints[i+1]] for i in range(len(breakpoints)-1)]
                    fwd_grad = param_evaler.nn_loss_grad(fwd, train_input, train_target)
                    # Split
                    fwd_grad_list = [fwd_grad[:, breakpoints[i]:breakpoints[i+1]] for i in range(len(breakpoints)-1)]
                    transformed_gradients = lmd_model.transform_gradients(fwd_grad_list)
                    for ctr, t_grad in enumerate(transformed_gradients):
                        params_primal[ctr] = params_primal[ctr] - lmd_model.stepsize[i] * t_grad
                    
                    fwd = torch.cat(params_primal, dim=1)

                    fwd_loss = param_evaler.nn_loss(fwd, train_input, train_target)
                avg_loss += fwd_loss.item()
                avg_end += test_acc(param_evaler.nn_fwd(fwd,test_input), test_target).item()
        to_divide = lmd_hparams["valid_num_tests"] * lmd_hparams["meta_batchsize"]
        logger.info("Epoch {}: begin_acc {:.5f}, end_acc {:.5f}, loss {:.5f}".format(str(meta_epoch), avg_start/to_divide, avg_end/to_divide, avg_loss/to_divide))
        print("Epoch {}: begin_acc {:.5f}, end_acc {:.5f}, loss {:.5f}".format(str(meta_epoch), avg_start/to_divide, avg_end/to_divide, avg_loss/to_divide))

        lmd_model.train()


    ### checkpointing
    if meta_epoch % lmd_hparams["checkpoint_freq"] == 0 and meta_epoch != 0:
        print("Checkpoint saved {}".format(str(meta_epoch)))
        torch.save(lmd_model.state_dict(), os.path.join(lmd_hparams['ckpt_path'], str(meta_epoch)))

