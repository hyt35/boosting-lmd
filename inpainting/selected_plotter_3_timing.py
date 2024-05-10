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
from matplotlib.lines import Line2D
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

figs_dir = "figs/inpaint_new"
datpath = "dat/inpaint"


# Stuff to plot:
stepsize_dict_plot = {
    "AMD": "recip",
    "ASMD": "mean",
    "SMD": "recip_x2",
    "MD": "recip_x2"
}
#%%
# Load models

#%%
# Initialize datasets and fbp operator


#%%

if __name__ == '__main__': 
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    true_fwd = torch.from_numpy(np.load(os.path.join(datpath, "true_min_arr.npy"))).to(device)
    # gdloss =  np.load(os.path.join(datpath, "true_gd_progression.npy"))
    true_min = np.load(os.path.join(datpath, "true_min_val.npy"))
    # nesterovloss = np.load(os.path.join(datpath, "nesterov_progression.npy"))
    # adamloss = np.load(os.path.join(datpath, "adam_progression.npy"))



    # banertloss = np.load('/local/scratch/public/hyt35/lmd_extensions_R1/banert_data/banert_stl10_inpaint.npy')

    # banertloss = np.insert(banertloss, 0,adamloss[0])
    # PLOT 1: different step size extensions for each opti algorithm

    # print(true_min)
    aux_function_names = ["time"]

    # fig, ax = plt.subplots(figsize = (8.0,4.8), dpi = 150) # loss plot
    # fig2, ax2 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # log-loss plot 
    # fig2, ax2 = plt.subplots(figsize = (6.0,4.0), dpi = 150) # log-loss plot 
    # fig3, ax3 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # l2 plot
    # fig4, ax4 = plt.subplots(figsize = (8.0,4.8), dpi = 150) # long log-loss
    figt, axt = plt.subplots(figsize = (8.0,4.8), dpi = 150) # long log-loss

    # fig.suptitle("Reconstruction Loss" )
    figt.suptitle("Reconstruction Loss")
    # fig3.suptitle("L2 distance")

    for gd_ss in ["2e-4", "5e-4", "1e-3", "2e-3", "5e-3", "1e-2", "2e-2", "5e-2", "1e-1"]:
        gdloss =  np.load(os.path.join(datpath, 'baseline2', "gd_progression"+gd_ss+".npy"))
        gdtime = np.load(os.path.join(datpath, 'baseline2', "gd_time"+gd_ss+".npy"))
        axt.plot(gdtime, gdloss[1:]-true_min, color='b', alpha=0.2)

    for nesterov_ss in ["1e-4", "2e-4", "5e-4", "1e-3", "2e-3", "5e-3"]:
        nesterovloss =  np.load(os.path.join(datpath, 'baseline2', "nesterov_progression_"+nesterov_ss+".npy"))
        nesterovtime = np.load(os.path.join(datpath, 'baseline2', "nesterov_time"+nesterov_ss+".npy"))
        axt.plot(nesterovtime, nesterovloss[1:]-true_min, color='orange', alpha=0.2)

    for adam_ss in ["1e-3", "2e-3", "5e-3", "1e-2", "2e-2", "5e-2", "1e-1"]:
        adamloss =  np.load(os.path.join(datpath, 'baseline2', "adam_progression"+adam_ss+".npy"))
        adamtime = np.load(os.path.join(datpath, 'baseline2', "adam_time"+adam_ss+".npy"))
        axt.plot(adamtime, adamloss[1:]-true_min, color='g', alpha=0.2)



    # nesterovloss = np.load(os.path.join(datpath, 'baseline', "nesterov_progression_2e-3.npy"))
    # adamloss = np.load(os.path.join(datpath, 'baseline', "adam_progression1e-2.npy"))
    # axt.plot(gdloss[:2000],  markevery=100, label="GD")
    # axt.plot(nesterovloss[:2000],  markevery=100, label="Nesterov")
    # axt.plot(adamloss[:2000],  markevery=100, label="Adam")

    # ax.plot(banertloss[:2000],  markevery=100, label="LPD")
    
    

    # ax2.loglog(range(1, 2001), gdloss[:2000]-true_min,  markevery=100, label="GD")
    # ax2.loglog(range(1, 2001), nesterovloss[:2000]-true_min,  markevery=100, label="Nesterov")
    # ax2.loglog(range(1, 2001), adamloss[:2000]-true_min,  markevery=100, label="Adam")
    # ax2.loglog(range(1, 2001), banertloss[:2000]-true_min,  markevery=100, label="LPD")
    # next(ax3._get_lines.prop_cycler)
    # next(ax3._get_lines.prop_cycler)
    # next(ax3._get_lines.prop_cycler)

    # ax4.loglog(range(1, len(gdloss)+1), gdloss-true_min,  markevery=100, label="GD")
    # ax4.loglog(range(1, len(nesterovloss)+1), nesterovloss-true_min,  markevery=100, label="Nesterov")
    # ax4.loglog(range(1, len(adamloss)+1), adamloss-true_min,  markevery=100, label="Adam")
    # ax4.loglog(range(1, len(banertloss)+1), banertloss-true_min,  markevery=100, label="LPD")
    # ax.set_xlabel("Iteration")
    # ax2.set_xlabel("Iteration")
    # ax3.set_xlabel("Iteration")
    # ax4.set_xlabel("Iteration")

    # ax.set_ylabel(r"$f(x^{(k)})$")
    # ax2.set_ylabel(r"$f(x^{(k)})-f^*$")
    # ax3.set_ylabel(r"$||x^{(k)}-x^*||$")
    # ax4.set_ylabel(r"$f(x^{(k)})-f^*$")

    # ax.set_ylim(top=gdloss[0]*1.5)
    # ax2.set_ylim(top=(gdloss[0]-true_min)*5)
    # ax3 set later
    # fooger = None
    # ax4.set_ylim(top=(gdloss[0]-true_min)*5)
    # PLOT 1: different step size extensions for each opti algorithm
    next(axt._get_lines.prop_cycler)
    next(axt._get_lines.prop_cycler)
    next(axt._get_lines.prop_cycler)
    next(axt._get_lines.prop_cycler)
    for opti_algo, opti_name in zip(["AMD", "SMD", "ASMD", "MD"], ["LAMD", "LSMD", "LASMD", "LMD"]):
    # for opti_algo, opti_name in zip([ "SMD","AMD"], ["LMD","LAMD"]):
    # for opti_algo, opti_name in zip(["SMD"], ["LMD"]):
    # for opti_algo in ["ASMD"]:
        # stepsize extension


        # subfolders in opti_algo folder
        # expects loss_arr.npy and other aux_fn.npy
        # for sub_type in ["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5"]:
        for sub_type in ["mean", "min", "final", "recip", "recip_power","recip_x2", "recip_x0.5"]:
            if sub_type != stepsize_dict_plot[opti_algo]:
                continue
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
            axt.plot(aux_arr, loss_array-true_min, '--', label=opti_name)
    
    axt.set_xlabel('Time (s)')
    axt.set_yscale('log')
    axt.set_ylim(top=1e4)
    axt.set_xlim(right=25)
    handles, labels = axt.get_legend_handles_labels()
    l1 = Line2D([0], [0], label='GD', color='b', alpha=0.3)
    l2 = Line2D([0], [0], label='Nesterov', color='orange', alpha=0.3)
    l3 = Line2D([0], [0], label='Adam', color='green', alpha=0.3)

    handles = [l1, l2, l3] + handles
    print(handles)
    axt.legend(handles=handles)
    figt.savefig(os.path.join(figs_dir, "vs_time"))
    figt.savefig(os.path.join(figs_dir, "vs_time.pdf"))
    # print(fooger)
    # ax3.set_ylim(bottom=0,top=fooger*1.2)
    # ax.legend()
    # # ax2.legend(prop={'size': 12})
    # ax2.legend()
    # ax3.legend()
    # ax4.legend()
    # save
    # fig.savefig(os.path.join(figs_dir, "losses_banert"))
    # fig2.savefig(os.path.join(figs_dir, "loglosses_banert"))
    # fig3.savefig(os.path.join(figs_dir, "l2_banert"))
    # fig4.savefig(os.path.join(figs_dir, "long_loglosses_banert"))
    # fig.savefig(os.path.join(figs_dir, "losses_banert.pdf"))
    # fig2.savefig(os.path.join(figs_dir, "loglosses_banert.pdf"))
    # fig3.savefig(os.path.join(figs_dir, "l2_banert.pdf"))
    # fig4.savefig(os.path.join(figs_dir, "long_loglosses_banert.pdf"))
# %%
