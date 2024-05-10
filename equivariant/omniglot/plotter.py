# plots data made from evaluate_omniglot_LMD_and_reptile
import matplotlib.pyplot as plt
import numpy as np
import os
# names = ['SGD', 'Adam', 'LMD', 'Reptile_SGD', 'Reptile_Adam', 'Reptile_LMD']
# bar = ['SGD', 'Adam', 'LMD', 'Reptile-SGD', 'Reptile-Adam', 'Reptile-LMD']
names = ['SGD', 'Adam', 'LMD', 'LAMD', 'Reptile_SGD', 'Reptile_Adam', 'Reptile_LMD', 'Reptile_LAMD']
bar = ['SGD', 'Adam', 'LMD', 'LAMD', 'Reptile-SGD', 'Reptile-Adam', 'Reptile-LMD', 'Reptile-LAMD']

losses_fig, losses_ax = plt.subplots(figsize = (8.0,4.8), dpi = 150)
accs_fig, accs_ax = plt.subplots(figsize = (8.0,4.8), dpi = 150)
taccs_fig, taccs_ax = plt.subplots(figsize = (8.0,4.8), dpi = 150)

flag = False
for prefix, foo in zip(names, bar):
    loss = np.load(os.path.join('exp_data', prefix, "loss.npy"))
    accs = np.load(os.path.join('exp_data', prefix, "accs.npy"))
    taccs = np.load(os.path.join('exp_data', prefix, "taccs.npy"))

    if "Reptile" in prefix:
        if not flag:
            losses_ax.set_prop_cycle(None)
            accs_ax.set_prop_cycle(None)
            taccs_ax.set_prop_cycle(None)
            flag = True
        losses_ax.plot(loss, '--', label = foo)
        accs_ax.plot(accs, '--', label = foo)
        taccs_ax.plot(taccs, '--', label = foo)
    else:
        losses_ax.plot(loss, label = foo)
        accs_ax.plot(accs, label = foo)
        taccs_ax.plot(taccs, label = foo)

losses_ax.set_yscale('log')
# Formattng
losses_fig.suptitle("Training negative log-likelihood")
accs_fig.suptitle("Train accuracy")
taccs_fig.suptitle("Test accuracy") 


losses_ax.set_xlabel("Iteration")
accs_ax.set_xlabel("Iteration")
taccs_ax.set_xlabel("Iteration")

losses_ax.set_ylabel("Cross-entropy Loss")
accs_ax.set_ylabel("Accuracy")
taccs_ax.set_ylabel("Accuracy")

accs_ax.set_ybound(upper=1.)
taccs_ax.set_ybound(upper=0.8)

losses_ax.legend(loc='lower left')
accs_ax.legend(loc='lower right')
taccs_ax.legend(loc='lower right')

losses_fig.savefig("figs/losses.png")
accs_fig.savefig("figs/accs.png")
taccs_fig.savefig("figs/taccs.png")

losses_fig.savefig("figs/losses.pdf")
accs_fig.savefig("figs/accs.pdf")
taccs_fig.savefig("figs/taccs.pdf")