import torch
# import utils, mirror_maps, generalization_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import torch.autograd as autograd
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from itertools import cycle
import os,time
import tqdm
import copy
import logging
from argparse import ArgumentParser
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from datetime import datetime


import warnings
warnings.filterwarnings("ignore") # necessary for torchmeta

torch.manual_seed(0)
np.random.seed(0)

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

class ParamEvaler(nn.Module):
    def __init__(self, batch_, label_, num_inits, size_array, model_instance ="CNNModel", model_mode=["conv", "conv", "dense"]):
        super().__init__()
        self.device = device
        self.lossFn = torch.nn.NLLLoss(reduction='mean')
        self.num_inits = num_inits
        self.model_mode = model_mode
        self.size_array = size_array
        self.model_instance = model_instance # "CNNModel"|"CNNModel5". affects the pooling later
        curr_batch = batch_.to(device) 
        curr_label = label_.to(device)
        if self.model_instance == "CNNModelSmall":
            foo = nn.MaxPool2d(2,2)
            curr_batch = foo(curr_batch)
        curr_batch.unsqueeze_(0)
        self.batch = curr_batch
        self.label = curr_label
        self.batch.requires_grad_(False)
        self.label.requires_grad_(False)

        flat_sizes = list(map(lambda x: np.prod(x), size_array))
        self.breakpoints = np.concatenate(([0],np.cumsum(flat_sizes)))


    def nn_fwd(self,params):
        curr_batch = self.batch.repeat(self.num_inits, 1,1,1,1)
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
            #####
            # check it is doing what it is supposed to 
            # foo = F.conv2d(out[0], weight[0], bias[0])
            #####
            if mode == "conv":

                # use groups to cheekily do conv2d in parallel
                # weight shape: [num_inits, C_out, C_in, k, k]
                # bias shape: [num_inits, C_out]
                curr_shape = out.shape # [num_inits, N, C_in, H, W]
                N = curr_shape[1]
                C_in = weight.shape[2]
                C_out = weight.shape[1]





                # hack the NN batch dimension into channels
                # to make use of group functionality
                # need to do some weird transpose to work correctly
                # https://discuss.pytorch.org/t/batched-conv2d-with-filters-grouped-in-batch-dimension/166861/2
                # +++++++++++++++++++++++++
                w_shape = weight.shape
                # weight = weight.reshape(self.num_inits * C_out, *w_shape[-3:])
                # bias = bias.reshape(self.num_inits * C_out)
                weight = weight.reshape(self.num_inits * C_out, *w_shape[-3:])
                bias = bias.flatten()
                # shape: [N, C_in * num_inits, H, W]
                out_fast = out.reshape(self.num_inits, N, C_in, *curr_shape[-2:]).transpose(0,1).reshape(N, self.num_inits * C_in, *curr_shape[3:])
                # print(out_fast.shape)
                # shape: [N, C_out*num_inits, H, W], hopefully



                # print("after reshape")
                # time.sleep(3)
                out_grouped = F.conv2d(out_fast, weight, bias, groups=self.num_inits)
                # print("after convv")
                # time.sleep(3)
                out_grouped_shape = out_grouped.shape
                # print(out_grouped_shape)
                # deinterleave
                out = out_grouped.reshape(N, self.num_inits,C_out, *out_grouped_shape[-2:]).transpose(0,1)
                # print("ooo", out.shape)
                out = F.relu(out)
                # +++++++++++++++++++++++++
                #####
                # print(torch.isclose(foo, out[0]))
                # raise Exception()
                #####



                # un-grouped vrsion
                # even worse
                # foo = torch.empty(self.num_inits, N, C_out, curr_shape[-2]-2, curr_shape[-1]-2).to(device)
                # for j in range(self.num_inits):
                #     foo[j] = F.conv2d(out[j], weight[j], bias[j])
                # out = F.relu(foo)

                
                # print("after relu")
                # time.sleep(3)

                out = pool_layer_3d(out)
                # out = pool_layer(out.reshape(-1, *out.shape[2:])) # [num_inits*N, C, H_new, W_new]
                # print("after pool")
                # time.sleep(3)
                # out = out.reshape((self.num_inits,N,)+out.shape[1:]) # reshape [num_inits, N, C, H, W]

            elif mode == "dense":
                if len(out.shape) == 5:
                    # flattening image, from [num_inits, N, C, H, W] -> [num_inits, N, C_new]
                    out = torch.flatten(out, start_dim=-3) 
                elif len(out.shape) != 3:
                    raise Exception("Unexpected shape")
                out = torch.matmul(out, torch.transpose(weight,-1,-2)) + bias[:,None,:] # 
                
            i = i+2

        out = F.log_softmax(out, dim=2)
        return out #shape (n_inits, batchsize, 10), hopefully


    def nn_loss(self,params):
        curr_label = self.label.repeat(self.num_inits)
        out = self.nn_fwd(params)
        return self.lossFn(out.view(-1,10), curr_label) * self.num_inits #/ hparams['train_sample_count'] # divide by moons batchsize
    
    def nn_loss_grad(self,layer_mat):
        return autograd.grad(self.nn_loss(layer_mat), layer_mat)[0]
    

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
# https://github.com/openai/supervised-reptile/tree/master

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

def update_params(base_model, new_model, stepsize):
    # updates base_model in place
    if type(base_model) != type(new_model):
        raise Exception("Different model type")
    
    for p, q in zip(base_model.parameters(), new_model.parameters()):
        with torch.no_grad():
            p.data += stepsize * (q-p)
    return base_model

def eval_model(model, test_loader):
    current_acc = 0.
    for batch_, targ_ in test_loader:
        batch, targ = batch_.to(device), targ_.to(device)
        pred = model(batch)
        current_acc += test_acc(pred, targ)

    return current_acc / len(test_loader)

def test_acc(pred, target):
    return torch.sum(torch.argmax(pred, dim=1) == target) / len(target)

def eval_reptile(model, train_loader, test_loader, hparams, logger = None):
    loss_fn = torch.nn.NLLLoss()
    dupe_model = copy.deepcopy(model)
    dupe_model.train()
    opt = torch.optim.SGD(dupe_model.parameters(), lr=hparams["opt_stepsize"])
    # eval before training
    acc_before_training = eval_model(dupe_model, test_loader)
    # if logger is not None:
    #     logger.info("acc before: {:.4f}".format(acc_before_training.item()))
    print("acc before: {:.4f}".format(acc_before_training.item()))
    # train section
    for _, (batch_, labels_) in zip(range(hparams['inner_iters']), cycle(train_loader)):
        batch, labels = batch_.to(device), labels_.to(device)
        loss = loss_fn(dupe_model(batch), labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
    # eval after training

    acc_after_training = eval_model(dupe_model, test_loader)
    if logger is not None:
        logger.info("before: {:.4f}, after {:.4f}".format(acc_before_training.item(), acc_after_training.item()))
    print("acc after: {:.4f}".format(acc_after_training.item()))

    return
    
        
def run_reptile(model, hparams, logger = None):
    dataset = omniglot("data", ways=10, shots=5, test_shots=15, meta_train=True, download=True)
    dataset_valid = omniglot("data", ways=10, shots=5, test_shots=15, meta_test=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=hparams["meta_batch"], num_workers=4)# print(len(dataloader))
    dataloader_valid = BatchMetaDataLoader(dataset_valid, batch_size=hparams["meta_batch"], num_workers=1)# print(len(dataloader))

    base_model = model
    # mnist_dataset = datasets.MNIST('/local/scratch/public/hyt35/ICNN-MLE/datasets', train=True,
    #     transform=transforms.Compose([transforms.ToTensor()]))
    # test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('/local/scratch/public/hyt35/ICNN-MLE/datasets', train=False,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         # transforms.Normalize((0.1307,), (0.3081,))
    #         ])
    # ), batch_size=1000)


    loss_fn = torch.nn.NLLLoss()
    

    # Meta loop -- performs reptile steps
    for meta_ctr, batch in tqdm.tqdm(zip(range(hparams['meta_iters']+1), dataloader), total=hparams['meta_iters']+1):
        # 1. Create copy of initialization model
        current_model = copy.deepcopy(model)
        # 2. Perform SGD on initialization model
        current_model.train()
        opt = torch.optim.SGD(current_model.parameters(), lr=hparams["opt_stepsize"])


        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]
        train_inputs, train_targets, test_inputs, test_targets = train_inputs.to(device), train_targets.to(device), test_inputs.to(device), test_targets.to(device)


        for task_idx, (train_input, train_target, test_input,
        test_target) in enumerate(zip(train_inputs, train_targets,
        test_inputs, test_targets)):

            for i in range(hparams["inner_iters"]):
                loss = loss_fn(current_model(train_input), train_target)

                opt.zero_grad()
                loss.backward()
                opt.step()


        # 3. Reptile update
        frac_done = meta_ctr / hparams['meta_iters']
        cur_meta_step_size = (1-frac_done) * hparams['meta_stepsize'] 
        update_params(base_model, current_model, cur_meta_step_size)

        # 4. (Optional) validation
        if meta_ctr % hparams['valid_freq'] == 0:
            start_acc, errs, end_acc = 0., 0., 0.

            for j in range(10):
                current_model = copy.deepcopy(model)
                current_model.train()


                opt = torch.optim.SGD(current_model.parameters(), lr=hparams["opt_stepsize"])
                batch_valid = next(iter(dataloader_valid))
                train_inputs, train_targets = batch_valid["train"]
                test_inputs, test_targets = batch_valid["test"]

                train_inputs, train_targets, test_inputs, test_targets = train_inputs.to(device), train_targets.to(device), test_inputs.to(device), test_targets.to(device)
                

                for task_idx, (train_input, train_target, test_input,
                test_target) in enumerate(zip(train_inputs, train_targets,
                test_inputs, test_targets)):

                    start_acc += test_acc(current_model(test_input), test_target).item()
                    # valid training phase
                    for i in range(hparams["eval_iters"]):
                        loss = loss_fn(current_model(train_input), train_target)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    # valid eval phase
                    end_acc += test_acc(current_model(test_input), test_target).item()
                    errs += loss.item()
            

            if logger is not None:
                logger.info('meta_ctr {}: begin_acc {:.5f}, end_acc {:.5f}, loss {:.5f}'.format(meta_ctr,
                start_acc/10, end_acc/10, errs/10))
            print('meta_ctr {}: begin_acc {:.5f}, end_acc {:.5f}, loss {:.5f}'.format(meta_ctr,
                start_acc/10, end_acc/10, errs/10))
                

        if meta_ctr % hparams['meta_checkpoint_freq'] == 0:
            # save dict
            torch.save(base_model.state_dict(), os.path.join(hparams['ckpt_path'],str(meta_ctr)))
    return 





if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='CNNsmall')
    parsed = parser.parse_args()
    mode = parsed.mode

    if mode != 'CNNsmall' and mode != '5layer':
        raise NotImplementedError()
    
    hparams = {'ckpt_path': 'ckpt/reptile',
               'meta_iters': 500000,
                'meta_stepsize': 0.1,
                'meta_batch': 1,
                'inner_batch': 100,
                # 'inner_sgd_batchsize': 10 , # full batch only
                'inner_iters': 10,
                'opt_stepsize': 1e-2 if mode == 'CNNsmall' else 1e-1,
                'meta_checkpoint_freq':5000,
                'valid_freq':500,
                'eval_batchsize':10,
                'eval_iters':100,
                'from_ckpt':-1
              }



    if mode == '5layer':
        init_model = FiveLayerDense().to(device)
    else:
        init_model = CNNModelSmall().to(device)

    if not os.path.exists(hparams["ckpt_path"]):
        os.makedirs(hparams["ckpt_path"])
    logger = setup_logger('logger', 'logs/log_reptile.log')
    now = datetime.now()
    logger.info("Start train " + now.strftime("%d-%m %H:%M"))
    run_reptile(init_model, hparams, logger)
