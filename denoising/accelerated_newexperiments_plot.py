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
COMPUTETRUEMIN = True

#%% INITIALIZATION
device0 = 'cuda:0'
device1 = 'cuda:1'

workdir_mom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_mom"
workdir_nomom = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/fbp_nomom"

if MODE == "ELLIPSE":
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/LMDExt_ellipse_new"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/LMDExt_datELLIPSE"

else:
    figs_dir = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/figs/LMDExt_lodopab"
    datpath = "/local/scratch/public/hyt35/icnn-md-mom/ellipse/LMDExt_datLODOPAB"



#%%
# Load models

#%%
# Initialize datasets and fbp operator


#%%

if __name__ == '__main__': 
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    true_fwd = torch.from_numpy(np.load(os.path.join(datpath, "true_min_arr.npy"))).to(device0)
    true_fwd1 = true_fwd.clone().to(device1)
    # gdloss =  np.load(os.path.join(datpath, "true_gd_progression.npy"))
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    # nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))
    # adamloss = np.load(os.path.join(datpath, "adam_progression.npy"))

    gdloss =  np.load(os.path.join(datpath, 'baselines', "gd_progression2e-3.npy"))
    nesterovloss = np.load(os.path.join(datpath, 'baselines', "nesterov_progression_1e-4.npy"))
    adamloss = np.load(os.path.join(datpath, 'baselines', "adam_progression1e-3.npy"))
    # print(true_min)
    aux_function_names = ["l2_tomin"]

    # PLOT 1: different step size extensions for each opti algorithm
    for opti_algo, opti_name in zip(["AMD", "SMD", "ASMD"], ["LAMD", "LSMD", "LASMD"]):
    # for opti_algo in ["ASMD"]:
        # stepsize extension
        fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
        fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
        fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 plot
        fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # long log-loss

        fig.suptitle("loss "+str(opti_algo))
        fig2.suptitle("Reconstruction Loss")
        fig3.suptitle("l2 distance " + str(opti_algo))
        fig4.suptitle("Reconstruction Loss")
        # ax.plot(gdloss[:500],  markevery=100, label="GD")
        # ax.plot(nesterovloss[:500],  markevery=100, label="Nesterov")
        # ax.plot(adamloss[:500],  markevery=100, label="Adam")

        # ax2.loglog(range(1, 2001), gdloss[:2000]-true_min,  markevery=100, label="GD")
        # ax2.loglog(range(1, 2001), nesterovloss[:2000]-true_min,  markevery=100, label="Nesterov")
        # ax2.loglog(range(1, 2001), adamloss[:2000]-true_min,  markevery=100, label="Adam")
        next(ax3._get_lines.prop_cycler)
        next(ax3._get_lines.prop_cycler)
        next(ax3._get_lines.prop_cycler)
        # ax4.loglog(range(1, len(gdloss)+1), gdloss-true_min,  markevery=100, label="GD")
        # ax4.loglog(range(1, len(nesterovloss)+1), nesterovloss-true_min,  markevery=100, label="Nesterov")
        # ax4.loglog(range(1, len(adamloss)+1), adamloss-true_min,  markevery=100, label="Adam")
        ax.set_xlabel("Iteration")
        ax2.set_xlabel("Iteration")
        ax3.set_xlabel("Iteration")
        ax4.set_xlabel("Iteration")

        ax.set_ylabel(r"$f(x^{(k)})$")
        ax2.set_ylabel(r"$f(x^{(k)})-f^*$")
        ax3.set_ylabel(r"$||x^{(k)}-x^*||$")
        ax4.set_ylabel(r"$f(x^{(k)})-f^*$")
        # subfolders in opti_algo folder
        # expects loss_arr.npy and other aux_fn.npy
        # for sub_type in ["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5"]:
        # for sub_type, sub_name in zip(["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5", "constant_one"]\
        #                              ,["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5", "constant_one"]):
        for sub_type, sub_name in zip(["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5"]\
                                    ,["Mean", "Min", "Final", r"$c/k$", r"$c'/\sqrt{k}$",r"$2c/k$", r"$c/2k$"]):
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
            ax.plot(loss_array,  markevery=100, linestyle = "dashed", label=sub_name)
            ax2.loglog(range(1, len(loss_array)+1),loss_array - true_min,  markevery=100, linestyle = "dashed", label=sub_name)
            ax3.plot(aux_arr,  markevery=100, linestyle = "dashed", label=sub_name)
            ax4.loglog(range(1, len(loss_array)+1),loss_array - true_min,  markevery=100, linestyle = "dashed", label=sub_name)
        ax.legend()
        ax2.legend()
        ax3.legend()
        ax4.legend()

        ax2.set_ylim(bottom=1e-1, top=1e4)
        # save
        fig.savefig(os.path.join(figs_dir_opti, "losses"))
        fig2.savefig(os.path.join(figs_dir_opti, "loglosses"))
        fig3.savefig(os.path.join(figs_dir_opti, "l2"))
        fig4.savefig(os.path.join(figs_dir_opti, "long_loglosses"))

        fig.savefig(os.path.join(figs_dir_opti, "losses.pdf"))
        fig2.savefig(os.path.join(figs_dir_opti, "loglosses.pdf"))
        fig3.savefig(os.path.join(figs_dir_opti, "l2.pdf"))
        fig4.savefig(os.path.join(figs_dir_opti, "long_loglosses.pdf"))
# %%
