import torch
import numpy as np
import os
import copy

class Perturber:
    def __init__(self, global_data):
        self.model = None
        self.perturbed_model = None
        self.global_data = global_data
        self.current_state_dict = None
        self.sigmas = None
    
    def load_model(self, model):
        self.model = model
        self.perturbed_model = copy.deepcopy(model)
        self.current_state_dict = copy.deepcopy(model.state_dict())

    
    def perturb(self, sigma):
        with torch.no_grad():
            for p, q in zip(self.model.parameters(), self.perturbed_model.parameters()):
                q.copy_(p + torch.randn_like(p)*sigma)
    
    def population_gradient(self, lossfn, model = None):
        if model is None:
            model = self.model
        model.zero_grad()
        if callable(self.global_data):
            global_dat, global_targ = self.global_data()
        else:
            global_dat, global_targ = self.global_data
        loss = lossfn(model(global_dat), global_targ)
        loss.backward()
        
        d_p_grad = []
        for p in model.parameters():
            if p.grad is not None:
                d_p_grad.append(p.grad.detach().clone())
        return d_p_grad

    def loss_gradient(self, lossfn, dat, targ, model = None):
        if model is None:
            model = self.model
        model.zero_grad()

        loss = lossfn(model(dat), targ)
        loss.backward()
        
        d_p_grad = []
        for p in model.parameters():
            if p.grad is not None:
                d_p_grad.append(p.grad)
        return d_p_grad
    
    def local_sensitivity(self, lossfn, dat, targ, sigma, n_samples = 50, minibatch_sampler = None):
        # Returns local value and gradient sensitivity
        # 
        if self.model is None:
            raise Exception("model not initialized")
        if callable(self.global_data):
            global_dat, global_targ = self.global_data()
        else:
            global_dat, global_targ = self.global_data

        loss = lossfn(self.model(dat), targ)

        pop_grad = self.population_gradient(lossfn)
        
        # Sum of perturbed terms
        # Divide by n_samples later to get expectation
        exp_perturbed_loss = 0. # sum E[L(w,s) - L(w+xi, s)]
        exp_perturbed_grad = 0. # sum E[||gbar(w) - gbar(w+xi) ||^2]
        for i in range(n_samples):
            self.perturb(sigma)
            loss_perturbed = lossfn(self.perturbed_model(dat), targ)

            pop_grad_perturbed = self.population_gradient(lossfn, model = self.perturbed_model)
            with torch.no_grad():
                for d_p, d_p_perturb in zip(pop_grad, pop_grad_perturbed):
                    exp_perturbed_grad += torch.sum(torch.pow(d_p - d_p_perturb, 2))
                exp_perturbed_loss += loss_perturbed

        # Expected perturbed loss
        exp_perturbed_loss /= n_samples
        exp_perturbed_grad /= n_samples

        local_val_sens = loss - exp_perturbed_loss
        local_grad_sens = exp_perturbed_grad

        # local gradient variance
        # requires a minibatcher E[||g(w, B_t) - gbar(w)||^2 | W_t = w]
        if minibatch_sampler is not None:
            exp_grad_variance = 0.
            def compute_loss(lossfn, dat_, targ_):
                grad_minibatch = self.loss_gradient(lossfn, dat_, targ_)
                # print("grad", grad_minibatch[1])
                exp_grad_variance_ = 0.
                with torch.no_grad():
                    for d_p, d_p_pop in zip(grad_minibatch, pop_grad): # why are these identical?
                        exp_grad_variance_ += torch.sum(torch.pow(d_p - d_p_pop, 2))
                return exp_grad_variance_

            # if callable(minibatch_sampler):
            #     for i in range(n_samples):
            #         dat_mini, targ_mini = minibatch_sampler()
            #         dat_mini = dat_mini.to(dat.device)
            #         targ_mini = targ_mini.to(dat.device)
            #         exp_grad_variance += compute_loss(lossfn, dat_mini, targ_mini)
            #     exp_grad_variance /= n_samples
            # else:
            #     for dat_mini, targ_mini in minibatch_sampler:
            #         exp_grad_variance += compute_loss(lossfn, dat_mini, targ_mini)
            #     exp_grad_variance /= len(minibatch_sampler)
            it = iter(minibatch_sampler)

            for i in range(n_samples):
                dat_mini, targ_mini = next(it)
                dat_mini = dat_mini.to(dat.device)
                targ_mini = targ_mini.to(dat.device)
                exp_grad_variance += compute_loss(lossfn, dat_mini, targ_mini)
            exp_grad_variance /= n_samples
            
            return local_val_sens, local_grad_sens, exp_grad_variance
        else:
            return local_val_sens, local_grad_sens
                
    def local_sensitivity_automagic(self, lossfn, dat, targ):
        # returns local value and gradient sensitivity
        # tracks the sigmas as well

        pass


@torch.no_grad()
def model_perturb(model, sigma, fpath = None):
    model_state_dict = copy.deepcopy(model.state_dict())
    if fpath is not None:
        torch.save(model_state_dict, fpath)
    for p in model.parameters():
        p += torch.randn_like(p)*sigma
    return model_state_dict

# TODO Local value sensitivity
def local_value_sensitivity(model, sigma, loss, data, target, n_samples = 50):
    # Evaluates \Delta_\sigma(w,s) = E[L(w,s) - L(w+\xi)+s]
    true_loss = loss(model(data), target)
    perturbed_loss = 0.
    # perturb the model
    for i in n_samples:
        old_model_state_dict = model_perturb(model, sigma)
        model.eval()
        with torch.no_grad():
            perturbed_loss += loss(model(data), target)
        model.load_state_dict(old_model_state_dict)
    model.train()

    perturbed_loss /= n_samples
    return true_loss - perturbed_loss

# TODO Local gradient sensitivity
def local_gradient_sensitivity(model, sigma, loss, data, target, n_samples = 50):
    pass
# TODO Local gradient variance