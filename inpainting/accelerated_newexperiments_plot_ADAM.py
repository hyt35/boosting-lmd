# -*- coding: utf-8 -*-
"""
Created on Fri Sep  13:44:37 2022

@author: hongy
"""
#from icnn import DenseICGN
#from denoising_nets_for_mcmc import ICNN

import numpy as np

import torch

import matplotlib.pyplot as plt
#from iunets import iUNet
import parse_import
import logging
from datetime import datetime
import torch.nn.functional as F
from dival import get_standard_dataset
from dival.datasets.fbp_dataset import get_cached_fbp_dataset
import tensorflow as tf

from pathlib import Path
import os

import time
import scipy.optimize as spopt
import odl

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


#%% INITIALIZATION
# device0 = 'cuda:0'
# device1 = 'cuda:1'
device = 'cuda'
# workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
# workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"

figs_dir = "figs/inpaint"
datpath = "dat/inpaint"



#%%
# Load models

#%%
# Initialize datasets and fbp operator


#%%

if __name__ == '__main__': 
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    true_fwd = torch.from_numpy(np.load(os.path.join(datpath, "true_min_arr.npy"))).to(device)
    gdloss =  np.load(os.path.join(datpath, "true_gd_progression.npy"))
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))
    adamloss = np.load(os.path.join(datpath, "adam_progression.npy"))
    # print(true_min)
    aux_function_names = ["l2_tomin"]

    # PLOT 1: different step size extensions for each opti algorithm
    for opti_algo in ["AMD", "SMD", "ASMD", "MD"]:
    # for opti_algo in ["ASMD"]:
        # stepsize extension
        fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
        fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
        fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 plot
        fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # long log-loss

        fig.suptitle("loss "+str(opti_algo))
        fig2.suptitle("logloss " + str(opti_algo))
        fig3.suptitle("l2 distance " + str(opti_algo))

        ax.plot(gdloss[:500],  markevery=100, label="gd")
        ax.plot(nesterovloss[:500],  markevery=100, label="nesterov")
        ax.plot(adamloss[:500],  markevery=100, label="adam")

        ax2.loglog(range(1, 501), gdloss[:500]-true_min,  markevery=100, label="gd")
        ax2.loglog(range(1, 501), nesterovloss[:500]-true_min,  markevery=100, label="nesterov")
        next(ax3._get_lines.prop_cycler)
        next(ax3._get_lines.prop_cycler)

        ax4.loglog(range(1, len(gdloss)+1), gdloss-true_min,  markevery=100, label="gd")
        ax4.loglog(range(1, len(nesterovloss)+1), nesterovloss-true_min,  markevery=100, label="nesterov")


        # subfolders in opti_algo folder
        # expects loss_arr.npy and other aux_fn.npy
        # for sub_type in ["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5"]:

        for sub_type in ["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5"]:
            figs_dir_opti = os.path.join(figs_dir, opti_algo)
            datpath_opti = os.path.join(datpath, opti_algo)

            figs_dir_step = os.path.join(figs_dir_opti, sub_type)
            datpath_step = os.path.join(datpath_opti, sub_type)


            # save to datpath
            # save data
            loss_array = np.load(os.path.join(datpath_step, "loss_arr.npy"))
            # loss_array[0] = 20000
            for aux_name in aux_function_names: # there is only one for now so it is fine
                aux_arr = np.load(os.path.join(datpath_step, aux_name+".npy"))
            # print(opti_algo, sub_type, loss_array[0])
            # bar = loss_array - true_min
            # print(opti_algo, sub_type, bar[0])
            ax.plot(loss_array,  markevery=100, linestyle = "dashed", label=sub_type)
            ax2.loglog(range(1, len(loss_array)+1),loss_array - true_min,  markevery=100, linestyle = "dashed", label=sub_type)
            ax3.plot(aux_arr,  markevery=100, linestyle = "dashed", label=sub_type)
            ax4.loglog(range(1, len(loss_array)+1),loss_array - true_min,  markevery=100, linestyle = "dashed", label=sub_type)

        ax.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()
        # save
        fig.savefig(os.path.join(figs_dir_opti, "losses"))
        fig2.savefig(os.path.join(figs_dir_opti, "loglosses"))
        fig3.savefig(os.path.join(figs_dir_opti, "l2"))
        fig4.savefig(os.path.join(figs_dir_opti, "long_loglosses"))