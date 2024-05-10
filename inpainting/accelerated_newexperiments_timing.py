# -*- coding: utf-8 -*-
"""
Created on Fri Sep  13:44:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN

import numpy as np
import torch
from torch._C import LoggerBase
import torch.nn as nn
import torch.autograd as autograd
import torchvision
import matplotlib.pyplot as plt
#from iunets import iUNet
import time
import tensorflow as tf

import os


from torchvision.utils import make_grid, save_image

torch.manual_seed(0)
np.random.seed(0)
device='cuda'
#%% EXPERIMENT PARAMETERS

# MAXITERS = 100 # maximum number of optim iter for experiments
PLOT = True # use the other script to plot.
#%% EXPERIMENT FLAGS

GRABNEW = True
COMPUTETRUEMIN = False

figs_dir = "figs/inpaint"
datpath = "dat/inpaint"
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
if not os.path.exists(datpath):
    os.makedirs(datpath)

#%% INITIALIZATION
checkpoint_path = '/local/scratch/public/hyt35/ICNN-MD/ICNN-STL10/checkpoints/Apr27_inpainting_denoising/780'
stl10_data = torchvision.datasets.STL10('/local/scratch/public/hyt35/datasets/STL10', split='test', transform=torchvision.transforms.ToTensor(), folds=1, download=True)


mask_ = torch.load('mask_80.pt') # mask with 20% remove
#mask = torch.load('mask_40.pt') # with 60% remove
mask = mask_.to(device)

def compute_gradient(net, inp):
    inp_with_grad = inp.requires_grad_(True)
    out = net(inp_with_grad)  
    fake = torch.cuda.FloatTensor(np.ones(out.shape)).requires_grad_(False)
    # Get gradient w.r.t. input
    gradients = autograd.grad(outputs=out, inputs=inp_with_grad,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    return gradients

class ICNN(nn.Module):
    def __init__(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5):
        super(ICNN, self).__init__()
        self.n_in_channels = num_in_channels
        self.n_layers = num_layers
        self.n_filters = num_filters
        self.kernel_size = kernel_dim
        self.padding = (self.kernel_size-1)//2
        #these layers should have non-negative weights
        self.wz = nn.ModuleList([nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
                                 for i in range(self.n_layers)])
        
        #these layers can have arbitrary weights
        self.wx_quad = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)\
                                 for i in range(self.n_layers+1)])
    
        self.wx_lin = nn.ModuleList([nn.Conv2d(self.n_in_channels, self.n_filters, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=True)\
                                 for i in range(self.n_layers+1)])
        
        #one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(self.n_filters, self.n_in_channels, self.kernel_size, stride=1, padding=self.padding, padding_mode='circular', bias=False)
        
        #slope of leaky-relu
        self.negative_slope = 0.2 
        self.strong_convexity = strong_convexity
        
        
    def scalar(self, x):
        z = torch.nn.functional.leaky_relu(self.wx_quad[0](x)**2 + self.wx_lin[0](x), negative_slope=self.negative_slope)
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(self.wz[layer](z) + self.wx_quad[layer+1](x)**2 + self.wx_lin[layer+1](x), negative_slope=self.negative_slope)
        z = self.final_conv2d(z)
        z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)
        
        return z_avg# + .5 * self.strong_convexity * (x ** 2).sum(dim=[1,2,3]).reshape(-1, 1)
    
    def forward(self, x):
        foo = compute_gradient(self.scalar, x)
        return (1-self.strong_convexity)*foo + self.strong_convexity*x
    
    #a weight initialization routine for the ICNN
    def initialize_weights(self, min_val=0.0, max_val=0.001, device=device):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val)\
            * torch.rand(self.n_filters, self.n_filters, self.kernel_size, self.kernel_size).to(device)
        
        self.final_conv2d.weight.data = min_val + (max_val - min_val)\
        * torch.rand(1, self.n_filters, self.kernel_size, self.kernel_size).to(device)
        return self
    
    #a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self): 
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        
        self.final_conv2d.weight.data.clamp_(0)
        return self 

class ICNNCouple(nn.Module):
    def __init__(self, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1)):
        super(ICNNCouple, self).__init__()
        self.fwd_model = None
        self.bwd_model = None
        self.stepsize = nn.Parameter(stepsize_init * torch.ones(num_iters).to(device))
        self.num_iters = num_iters
        self.ssmin = stepsize_clamp[0]
        self.ssmax = stepsize_clamp[1]
        # Logger
        # Checkpoint
        
    def init_fwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5):
        self.fwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity).to(device)
        return self
        
    def init_bwd(self, num_in_channels=1, num_filters=64, kernel_dim=5, num_layers=10, strong_convexity = 0.5):
        self.bwd_model = ICNN(num_in_channels, num_filters, kernel_dim, num_layers, strong_convexity).to(device)
        return self
    
    def clip_fwd(self):
        self.fwd_model.zero_clip_weights()
        return self
        
    def clip_bwd(self):
        self.bwd_model.zero_clip_weights()
        return self
        
    def forward(self, x, gradFun = None):
        if gradFun is None:
            raise RuntimeError("Gradient function not provided")
        with torch.no_grad():
            fwd = x
            for ss in self.stepsize:
                fwd = self.bwd_model(self.fwd_model(x) - ss*gradFun(fwd)) 
        return fwd
    
    def clamp_stepsizes(self):
        with torch.no_grad():
            self.stepsize.clamp_(self.ssmin,self.ssmax)
        return self
            
    def fwdbwdloss(self, x):
        return torch.linalg.vector_norm(self.bwd_model(self.fwd_model(x))-x, ord=1)


#%%
# Initialize models

icnn_couple = ICNNCouple(stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
icnn_couple.init_fwd(num_in_channels=3, num_filters=60, kernel_dim=3, num_layers=3, strong_convexity = 0.1)
icnn_couple.fwd_model.initialize_weights()
icnn_couple.init_bwd(num_in_channels=3, num_filters=75, kernel_dim=3, num_layers=5, strong_convexity = 0.5)
icnn_couple.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))

# save variable for later
adap_ss = icnn_couple.stepsize

#%%
# Initialize datasets and fbp operator

#%% parameter
noise_level = 0.1
reg_param = 0.15
stepsize = 0.01

#%%
# resize_op = torchvision.transforms.Resize([128,128])
if __name__ == '__main__': 
    icnn_couple.eval()
    # fig, ax = plt.subplots(figsize = (8,6)) # loss plot
    # fig2, ax2 = plt.subplots(figsize = (8,6)) # fwdbwd plot
    #%% check whether to get new data sample
    if GRABNEW:
        bsize=1

        test_dataloader = torch.utils.data.DataLoader(stl10_data, batch_size=bsize) # When training

        for idx, (batch_, _) in enumerate(test_dataloader):
            # add gaussian noise
            batch_masked_ = batch_*mask_ + 0.05*torch.randn_like(batch_)
            batch_masked = batch_masked_.to(device)

            # define objective functions
            def recon_err(img):
                tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
                tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
                tv = (tv_h+tv_w)
                fidelity = torch.pow((img-batch_masked)*mask,2).sum()
                return (fidelity + reg_param*tv)/2
            
            def recon_err_grad(img):
                return autograd.grad(recon_err(img), img)[0]
            break
    init_err = recon_err(batch_masked).item()
    init_closeness_mom = icnn_couple.fwdbwdloss(batch_masked).item()
    print("init complete")

    #%%
    if COMPUTETRUEMIN:
        # perform lots of gd
        gdloss = [init_err]
        fwd = batch_masked.clone().detach().requires_grad_(True)
        for i in range(15000):
            fwd = fwd.detach().requires_grad_(True)
            fwdGrad = recon_err_grad(fwd)
            fwd = fwd - 5e-4*fwdGrad
            gdloss.append(recon_err(fwd).item())
        for i in range(5000):
            fwd = fwd.detach().requires_grad_(True)
            fwdGrad = recon_err_grad(fwd)
            fwd = fwd - 1e-4*fwdGrad
            gdloss.append(recon_err(fwd).item())
        # plot
        true_min = recon_err(fwd).item()
        np.save(os.path.join(datpath, "true_gd_progression"), gdloss)
        np.save(os.path.join(datpath, "true_min_val"), true_min)
        np.save(os.path.join(datpath, "true_min_arr"), fwd.detach().cpu().numpy())
        print("compute true min", true_min)

        with tf.io.gfile.GFile(os.path.join(figs_dir, "true_recon_gd.png"), "wb") as fout:
            save_image(fwd.detach().cpu(), fout, nrow = 5)
        nesterovloss = [init_err]
        lam = 0
        currentstep = 1
        yk = batch_masked.clone().detach().requires_grad_(True)
        xk = yk
        for i in range(5000):
            yk = yk.detach().requires_grad_(True)
            xklast = xk
            xk = yk - 5e-4*recon_err_grad(yk)
            yk = xk + i/(i+3)*(xk-xklast)

            nesterovloss.append(recon_err(yk).item())

        np.save(os.path.join(datpath, "nesterov_progression"), nesterovloss)
    
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    true_fwd = torch.from_numpy(np.load(os.path.join(datpath, "true_min_arr.npy"))).to(device)
    init_l2 = torch.linalg.vector_norm(true_fwd - batch_masked).item()



    # %%
    def construct_extension(stepsizes, final_length, extend_type, p = None):
        """
        Generates extensions of stepsizes

        Args:
            stepsizes (Tensor): Initial stepsizes to extend
            final_length (int): Length of final stepsizes
            extend_type (string or float): method of extension, can be "max", "mean", "min", "final", "recip", or float quartile


        Returns:
            Tensor: Tensor of extended stepsizes, beginning with input stepsizes
        """
        num_to_extend = final_length - len(stepsizes)
        if num_to_extend < 0:
            raise Exception("final length less than number of steps")
        if extend_type == "max":
            to_cat = torch.ones(num_to_extend, device = stepsizes.device) * torch.max(stepsizes)
        elif extend_type == "mean":
            to_cat = torch.ones(num_to_extend, device = stepsizes.device) * torch.mean(stepsizes)
        elif extend_type == "min":
            to_cat = torch.ones(num_to_extend, device = stepsizes.device) * torch.min(stepsizes)
        elif extend_type == "final":
            to_cat = torch.ones(num_to_extend, device = stepsizes.device) * stepsizes[-1]
        elif extend_type == "recip": # ck^{-1}
            to_cat = torch.ones(num_to_extend, device = stepsizes.device)
            # first calculate c: taken as mean
            c = torch.mean(torch.arange(start=1, end = len(stepsizes)+ 1, device = stepsizes.device) * stepsizes)
            # then take c/k to concatenate
            to_cat = c/torch.arange(start = len(stepsizes)+ 1, end = final_length+1, device = stepsizes.device) 
        elif extend_type == "recip_power" and p is not None: # ck^{-p}
            to_cat = torch.ones(num_to_extend, device = stepsizes.device)
            # first calculate c: taken as mean
            c = torch.mean(torch.arange(start=1, end = len(stepsizes)+ 1, device = stepsizes.device)**p * stepsizes)
            # then take c/k to concatenate
            to_cat = c/(torch.arange(start = len(stepsizes)+ 1, end = final_length+1, device = stepsizes.device)**p)
        elif extend_type == "constant" and p is not None:
            to_cat = torch.ones(num_to_extend, device = stepsizes.device) * p
        else:
            raise Exception("Extension type not supported")
        
        extended_stepsizes = torch.cat((stepsizes, to_cat))
        return extended_stepsizes

    # for opti_algo in ["AMD"]:
    for opti_algo in ["AMD", "SMD", "ASMD", "MD"]:
        # define parameters
        dev = device
        max_iter = 2001
        if opti_algo == "AMD":
            # max_iter = 501
            r = 3
            gamma = 1 # maybe this should be increased, and the stepsizes correspondingly decreased
            # since gamma = 1 may not satisfy gamma >= L_R L_{Psi*}
            # since strong convexity of Psi is 0.1, strong smoothness modulus is at most 10 for L_{Psi*
            # L_R = 1 for R(x,y) = 1/2 ||x-y||^2
            
            def perform_optimization(data, model, stepsizes, f = recon_err, gradf = recon_err_grad, r = r, aux_functions = []):
                aux_arrays = []
                for aux_fn in aux_functions:
                    aux_arrays.append([])
                loss_array = []

                xktilde = data
                zktilde = data
                dual_zktilde = model.fwd_model(zktilde)


                k=0
                for ss in stepsizes:
                    lambda_k = r/(r+k)
                    x = lambda_k * zktilde + (1-lambda_k) * xktilde
                    x = x.detach()
                    
                    loss = f(x)
                    loss_array.append(loss.item())
                    for aux_fn, aux_index in zip(aux_functions, range(len(aux_functions))): 
                        try:
                            aux_arrays[aux_index].append(aux_fn(x).item())
                        except: 
                            aux_arrays[aux_index].append(aux_fn(x))

                    x.requires_grad_()
                    gradx = gradf(x)

                    dual_zktilde = dual_zktilde - k*ss/r * gradx
                    zktilde = model.bwd_model(dual_zktilde)
                    xktilde = x - gamma * ss * gradx


                    k += 1
                return x, loss_array, aux_arrays

        elif opti_algo == "SMD":
            # max_iter = 501
            noise_level = 0.0
            def perform_optimization(data, model, stepsizes, f = recon_err, gradf = recon_err_grad, sigma = noise_level, aux_functions = []):
                aux_arrays = []
                for aux_fn in aux_functions:
                    aux_arrays.append([])
                loss_array = []

                x = data
                dual_x = model.fwd_model(data)

                averaged_x = torch.zeros_like(x) # TODO

                for ss in stepsizes:
                    x = x.detach()
                    
                    loss = f(x)
                    loss_array.append(loss.item())
                    for aux_fn, aux_index in zip(aux_functions, range(len(aux_functions))): 
                        try:
                            aux_arrays[aux_index].append(aux_fn(x).item())
                        except: 
                            aux_arrays[aux_index].append(aux_fn(x))

                    x.requires_grad_()
                    gradx = gradf(x) + sigma * torch.randn_like(x)

                    dual_x = dual_x - ss * gradx
                    x = model.bwd_model(dual_x)

                return x, loss_array, aux_arrays
            

        elif opti_algo == "ASMD":
            # max_iter = 501
            noise_level = 0.0
            def perform_optimization(data, model, stepsizes, f = recon_err, gradf = recon_err_grad, sigma = noise_level, aux_functions = []):
                aux_arrays = []
                for aux_fn in aux_functions:
                    aux_arrays.append([])
                loss_array = []

                x = data
                dual_y = model.fwd_model(data)

                
                Ak = 1/2
                sk = 1/2
                k = 0
                for ss in stepsizes:
                    Ak1 = (k+1)*(k+2)/2
                    tauk = (Ak1-Ak)/Ak
                    sk = (k+1)**(3/2)

                    x = x.detach()
                    
                    loss = f(x)
                    loss_array.append(loss.item())
                    for aux_fn, aux_index in zip(aux_functions, range(len(aux_functions))): 
                        try:
                            aux_arrays[aux_index].append(aux_fn(x).item())
                        except: 
                            aux_arrays[aux_index].append(aux_fn(x))

                    x.requires_grad_()
                    gradx = gradf(x) + sigma * torch.randn_like(x)

                    dual_y = dual_y - ((Ak1 - Ak)/sk)*ss * gradx
                    x = model.bwd_model(dual_y)*tauk/(tauk+1)+ x/(tauk+1)

                    k += 1
                    Ak = Ak1
                return x, loss_array, aux_arrays

        elif opti_algo == "MD":
            # max_iter = 501
            def perform_optimization(data, model, stepsizes, f = recon_err, gradf = recon_err_grad, sigma = noise_level, aux_functions = []):
                aux_arrays = []
                for aux_fn in aux_functions:
                    aux_arrays.append([])
                loss_array = []

                x = data

                for ss in stepsizes:
                    x = x.detach()
                    
                    loss = f(x)
                    loss_array.append(loss.item())
                    for aux_fn, aux_index in zip(aux_functions, range(len(aux_functions))): 
                        try:
                            aux_arrays[aux_index].append(aux_fn(x).item())
                        except: 
                            aux_arrays[aux_index].append(aux_fn(x))

                    
                    x.requires_grad_()
                    gradx = gradf(x) + sigma * torch.randn_like(x)

                    dual_x = model.fwd_model(x) - ss * gradx
                    x = model.bwd_model(dual_x)

                return x, loss_array, aux_arrays



        figs_dir_opti = os.path.join(figs_dir, opti_algo)
        if not os.path.exists(figs_dir_opti):
            os.makedirs(figs_dir_opti)
        datpath_opti = os.path.join(datpath, opti_algo)
        if not os.path.exists(datpath_opti):
            os.makedirs(datpath_opti)


        aux_fn_l2 = lambda x: torch.linalg.vector_norm(x - true_fwd)
        aux_functions = [aux_fn_l2]
        aux_function_names = ["l2_tomin"]
        # stepsize extension
        for extend_type in ["mean", "min", "final", "recip", "recip_power", "recip_x2", "recip_x0.5", "constant_one"]:


            
            figs_dir_step = os.path.join(figs_dir_opti, extend_type)
            if not os.path.exists(figs_dir_step):
                os.mkdir(figs_dir_step)
            datpath_step = os.path.join(datpath_opti, extend_type)
            if not os.path.exists(datpath_step):
                os.mkdir(datpath_step)



            print("Start ", opti_algo, extend_type)
            start = time.time()
            def aux_fn_time(*args):
                return time.time() - start
            aux_functions = [aux_fn_l2, aux_fn_time]
            aux_function_names = ["l2_tomin", "time"]

            if extend_type == "recip_power":
                stepsizes = construct_extension(adap_ss, final_length=max_iter, extend_type = extend_type, p=0.5)
            elif extend_type == "recip_x2":
                stepsizes = construct_extension(adap_ss, final_length=max_iter, extend_type = "recip")*2
            elif extend_type == "recip_x0.5":
                stepsizes = construct_extension(adap_ss, final_length=max_iter, extend_type = "recip")*0.5
            elif extend_type == "constant_one": # For ASMD
                stepsizes = torch.ones(max_iter)
            else:
                stepsizes = construct_extension(adap_ss, final_length=max_iter, extend_type = extend_type)
            data = batch_masked.clone().detach().requires_grad_(True)
            final, loss_array, aux_arrays = perform_optimization(data, icnn_couple, stepsizes, aux_functions = aux_functions)

            print(loss_array[0],loss_array[50], loss_array[100], loss_array[150], loss_array[200], loss_array[500] )
            # save to datpath
            # save data
            np.save(os.path.join(datpath_step, "loss_arr"), np.array(loss_array))
            for aux_arr, aux_name in zip(aux_arrays, aux_function_names):
                np.save(os.path.join(datpath_step, aux_name), np.array(aux_arr))

            with tf.io.gfile.GFile(os.path.join(figs_dir_step, "final.png"), "wb") as fout:
                    save_image(final.detach().cpu(), fout, nrow = 5)

#%%