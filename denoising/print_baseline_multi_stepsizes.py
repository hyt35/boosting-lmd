# -*- coding: utf-8 -*-
"""
Created on Fri Sep  13:44:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN
from sqlite3 import TimeFromTicks
from xxlimited import foo
import numpy as np
from odl.tomo.analytic.filtered_back_projection import fbp_op
import torch
from torch._C import LoggerBase
import torch.nn as nn
import torch.autograd as autograd
import torchvision
import matplotlib.pyplot as plt
#from iunets import iUNet
import parse_import
import logging
from datetime import datetime
import torch.nn.functional as F
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
import tensorflow as tf
from models import ICNNCoupleMomentum, ICNNCouple
from pathlib import Path
import os
from tqdm import tqdm
import torch_wrapper
import time
import scipy.optimize as spopt
import odl
from dival.datasets.standard import get_standard_dataset
from torchvision.utils import make_grid, save_image
from odl import uniform_discr
from dival.util.odl_utility import ResizeOperator
torch.manual_seed(0)
np.random.seed(0)

#%% EXPERIMENT PARAMETERS
IMPL = 'astra_cuda'
MODE = "ELLIPSE"
# MODE = "LODOPAB"
EXPTYPE = "MAN" # experiment type 
MAXITERS = 100 # maximum number of optim iter for experiments
PLOT = True # use the other script to plot.
#%% EXPERIMENT FLAGS

GRABNEW = False
COMPUTETRUEMIN = False

#%% INITIALIZATION
device0 = 'cuda:0'
# device1 = 'cuda:1'

workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"

checkpoint_dir_mom = os.path.join(workdir_mom, "checkpoints", "20")
checkpoint_dir_nomom = os.path.join(workdir_nomom, "checkpoints", "20")

if MODE == "ELLIPSE":
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/LMDExt_ellipse"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/LMDExt_datELLIPSE"

else:
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/LMDExt_lodopab"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/LMDExt_datLODOPAB"
if not os.path.exists(figs_dir):
    os.mkdir(figs_dir)
if not os.path.exists(datpath):
    os.mkdir(datpath)

# args=parse_import.parse_commandline_args()

# #%%
# # Load models

# # icnn_couple_mom = ICNNCoupleMomentum(device = device0, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
# # icnn_couple_mom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
# # icnn_couple_mom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
# # icnn_couple_mom.load_state_dict(torch.load(checkpoint_dir_mom, map_location=device0)) # use this one
# # icnn_couple_mom.device = device0

# # icnn_couple_nomom = ICNNCouple(device = device1, stepsize_init = 0.01, num_iters = 10, stepsize_clamp = (0.001,0.1))
# # icnn_couple_nomom.init_fwd(num_in_channels=1, num_filters=60, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
# # icnn_couple_nomom.init_bwd(num_in_channels=1, num_filters=70, kernel_dim=3, num_layers=2, strong_convexity = 0.1, dense_size = 20)
# # icnn_couple_nomom.load_state_dict(torch.load(checkpoint_dir_nomom, map_location = device1))
# # icnn_couple_nomom.device = device1

# # save variable for later
# # adap_ss_mom = icnn_couple_mom.stepsize
# # adap_ss_nomom = icnn_couple_nomom.stepsize
# #%%
# # Initialize datasets and fbp operator

# #%% parameter
# noise_level = 0.1
# reg_param = 0.15
# stepsize = 0.01

# #%%
# resize_op = torchvision.transforms.Resize([128,128])
if __name__ == '__main__': 
    # icnn_couple_mom.eval()
    # icnn_couple_nomom.eval()
    # fig, ax = plt.subplots(figsize = (8,6)) # loss plot
    # fig2, ax2 = plt.subplots(figsize = (8,6)) # fwdbwd plot
    #%% check whether to get new data sample
    # if GRABNEW:
    #     bsize=5

    #     if MODE == "ELLIPSE":
    #         dataset = get_standard_dataset('ellipses', impl=IMPL)
    #         CACHE_FILES = {
    #             'train':
    #                 ('./cache_ellipses_train_fbp.npy', None),
    #             'validation':
    #                 ('./cache_ellipses_validation_fbp.npy', None)}

    #         def fbp_postprocess(fbp): # no correction needed for ellipses dataset
    #             return fbp
    #     else:
    #         dataset = get_standard_dataset('lodopab', impl=IMPL)
    #         CACHE_FILES = {
    #             'train':
    #                 ('./cache_lodopab_train_fbp.npy', None),
    #             'validation':
    #                 ('./cache_lodopab_validation_fbp.npy', None)}

    #         def fbp_postprocess(fbp): # correction for lodopab fbp inversion issue
    #             foo = fbp.view(fbp.shape[0], -1)
    #             min_ = torch.min(foo, dim = 1)[0]
    #             max_ = torch.max(foo, dim = 1)[0]

    #             min__ = min_[:,None,None,None]
    #             max__ = max_[:,None,None,None]
    #             fbp = (fbp-min__)/(max__-min__)
    #             return fbp

    #     # operators and dataset
    #     ray_trafo = dataset.get_ray_trafo(impl=IMPL)
    #     fbp_op_odl = odl.tomo.fbp_op(ray_trafo)
    #     fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device0)
    #     cached_fbp_dataset = get_cached_fbp_dataset(dataset, ray_trafo, CACHE_FILES)
    #     dataset.fbp_dataset = cached_fbp_dataset

        # dataset_train = dataset.create_torch_dataset(
        #     part='train', reshape=((1,) + dataset.space[0].shape,
        #                             (1,) + dataset.space[1].shape))

        # dataset_validation = dataset.create_torch_dataset(
        #     part='validation', reshape=((1,) + dataset.space[0].shape,
        #                                 (1,) + dataset.space[1].shape))

        # train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=bsize)

        # test_dataloader = torch.utils.data.DataLoader(dataset_validation, batch_size = bsize)
        # for idx, (batch_, gt) in enumerate(test_dataloader):
        #     gt = resize_op(gt)
        #     # add gaussian noise
        #     batch = batch_.to(device0)
        #     batch_noisy = batch + noise_level*torch.randn_like(batch) # 10% gaussian noise

        #     fbp_batch_noisy = resize_op(fbp_postprocess(fbp_op(batch))) # apply fbp to noisy
    #         break
    #     torch.save(gt, os.path.join(datpath, MODE+"gt.pt"))
    #     torch.save(fbp_batch_noisy, os.path.join(datpath, MODE+"batch.pt"))
    # else:
    #     gt = torch.load(os.path.join(datpath, MODE+"gt.pt")).to(device0)
    #     fbp_batch_noisy = torch.load(os.path.join(datpath, MODE+"batch.pt")).to(device0)


    #     with tf.io.gfile.GFile(os.path.join(figs_dir, "gt.png"), "wb") as fout:
    #         save_image(gt.detach().cpu(), fout, nrow = 5)
    #     with tf.io.gfile.GFile(os.path.join(figs_dir, "fbp_batch_noisy.png"), "wb") as fout:
    #         save_image(fbp_batch_noisy.detach().cpu(), fout, nrow = 5)
    
    
    
    
    
#     # define objective functions (on respective devices)
#     def recon_err(img):
#         tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
#         tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
#         tv = (tv_h+tv_w)
#         fidelity = torch.pow((img-fbp_batch_noisy),2).sum()
#         return (fidelity + reg_param*tv)/2

#     def recon_err_grad(img):
#         return autograd.grad(recon_err(img), img)[0]
#     # # def recon_err1(img):
#     # #     tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
#     # #     tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
#     # #     tv = (tv_h+tv_w)
#     # #     fidelity = torch.pow((img-fbp_batch_noisy1),2).sum()
#     # #     return (fidelity + reg_param*tv)/2
    
#     # def recon_err_grad1(img):
#     #     return autograd.grad(recon_err1(img), img)[0]
    
#     init_err = recon_err(fbp_batch_noisy).item()
#     print("init complete")



    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))


    
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150)
    fig2.suptitle("Reconstruction Loss")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$f(x^{(k)})-f^*$")

    for nesterov_ss in ["1e-4", "2e-4", "5e-4", "1e-3", "2e-3", "5e-3"]:
        loss_array = np.load(os.path.join(datpath, "baselines", "nesterov_progression_"+nesterov_ss+".npy"))
        ax2.loglog(range(1, len(loss_array)),loss_array[1:] - true_min,  markevery=100, label=nesterov_ss)
    ax2.legend()
    ax2.axvline(2000, alpha=0.5, color='grey')
    fig2.savefig(os.path.join(figs_dir, "baseline", "nesterov.pdf"))
    plt.close("all")

    ##########
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150)
    fig2.suptitle("Reconstruction Loss")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$f(x^{(k)})-f^*$")

    for adam_ss in ["1e-3", "2e-3", "5e-3", "1e-2", "2e-2", "5e-2", "1e-1"]:
        loss_array = np.load(os.path.join(datpath, "baselines", "adam_progression"+adam_ss+".npy"))
        ax2.loglog(range(1, len(loss_array)),loss_array[1:] - true_min,  markevery=100, label=adam_ss)
    ax2.legend()
    ax2.axvline(2000, alpha=0.5, color='grey')
    fig2.savefig(os.path.join(figs_dir, "baseline", "adam.pdf"))
    plt.close("all")
    #########
    fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150)
    fig2.suptitle("Reconstruction Loss")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"$f(x^{(k)})-f^*$")

    for gd_ss in ["2e-4", "5e-4", "1e-3", "2e-3", "5e-3", "1e-2", "2e-2", "5e-2", "1e-1"]:
        loss_array = np.load(os.path.join(datpath, "baselines", "gd_progression"+gd_ss+".npy"))
        ax2.loglog(range(1, len(loss_array)),loss_array[1:] - true_min,  markevery=100, label=gd_ss)
    ax2.legend()
    ax2.axvline(2000, alpha=0.5, color='grey')
    fig2.savefig(os.path.join(figs_dir, "baseline", "gd.pdf"))
    plt.close("all")

    # for adam_ss in ["1e-3", "2e-3", "5e-3", "1e-2", "2e-2", "5e-2", "1e-1"]:
    #     adamloss = [init_err]
    #     fwd = fbp_batch_noisy.clone().detach().requires_grad_(True)
    #     fwd = nn.Parameter(fwd)
    #     opt = torch.optim.Adam([fwd], lr=float(adam_ss))

    #     for i in range(5000):
    #         loss = recon_err(fwd)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #         adamloss.append(loss.item())
    #     print("Adam {} final {:.6f}".format(adam_ss, adamloss[-1]))
    #     np.save(os.path.join(datpath, "baselines", "adam_progression"+str(adam_ss)), adamloss)

    # for gd_ss in ["2e-4", "5e-4", "1e-3", "2e-3", "5e-3", "1e-2", "2e-2", "5e-2", "1e-1"]:
    #     gdloss = [init_err]
    #     fwd = fbp_batch_noisy.clone().detach().requires_grad_(True)
    #     fwd = nn.Parameter(fwd)
    #     opt = torch.optim.SGD([fwd], lr=float(gd_ss))

    #     for i in range(5000):
    #         loss = recon_err(fwd)
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #         gdloss.append(loss.item())
    #     print("GD{} final {:.6f}".format(gd_ss, gdloss[-1]))
    #     np.save(os.path.join(datpath, "baselines", "gd_progression"+str(gd_ss)), gdloss)

'''
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

    for opti_algo in ["ASMD"]:
    # for opti_algo in ["AMD", "SMD", "ASMD"]:
        # define parameters
        dev = device0
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
            noise_level = 0.05
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
            noise_level = 0.05
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




        figs_dir_opti = os.path.join(figs_dir, opti_algo)
        if not os.path.exists(figs_dir_opti):
            os.mkdir(figs_dir_opti)
        datpath_opti = os.path.join(datpath, opti_algo)
        if not os.path.exists(datpath_opti):
            os.mkdir(datpath_opti)


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
            if extend_type == "recip_power":
                stepsizes = construct_extension(adap_ss_mom, final_length=max_iter, extend_type = extend_type, p=0.5)
            elif extend_type == "recip_x2":
                stepsizes = construct_extension(adap_ss_mom, final_length=max_iter, extend_type = "recip")*2
            elif extend_type == "recip_x0.5":
                stepsizes = construct_extension(adap_ss_mom, final_length=max_iter, extend_type = "recip")*0.5
            elif extend_type == "constant_one": # For ASMD
                stepsizes = torch.ones(max_iter)
            else:
                stepsizes = construct_extension(adap_ss_mom, final_length=max_iter, extend_type = extend_type)
            data = fbp_batch_noisy.clone().detach().requires_grad_(True)
            final, loss_array, aux_arrays = perform_optimization(data, icnn_couple_mom, stepsizes, aux_functions = aux_functions)

            print(loss_array[0],loss_array[50], loss_array[100], loss_array[150], loss_array[200], loss_array[500] )
            # save to datpath
            # save data
            np.save(os.path.join(datpath_step, "loss_arr"), np.array(loss_array))
            for aux_arr, aux_name in zip(aux_arrays, aux_function_names):
                np.save(os.path.join(datpath_step, aux_name), np.array(aux_arr))

            with tf.io.gfile.GFile(os.path.join(figs_dir_step, "final.png"), "wb") as fout:
                    save_image(final.detach().cpu(), fout, nrow = 5)

#%%
'''