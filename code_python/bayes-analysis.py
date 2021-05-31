######### import packages #########
#import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt 
import pymc3 as pm
import seaborn as sns
import theano 
import arviz as az

### load data ###
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

###### plots ########
# NB: title?
sns.lmplot(x = "t", y = "y", hue = "idx", data = train)

##### m0 : no random effects, estimate all ###### 
def model_zero():
    
    with pm.Model() as m0: 
        
        # shared variables 
        t_shared = pm.Data('t_shared', t)
        idx_shared = pm.Data('idx_shared', idx)
        
        # specify priors  
        beta = pm.Normal("beta", mu = 0, sigma = 0.3, shape = n_id)
        alpha = pm.Normal("alpha", mu = 0, sigma = 0.5, shape = n_id)
        eps = pm.HalfNormal("eps", sigma = 0.5) # this is explicitly half in pymc3

        # calculate mu
        mu = alpha[idx_shared] + beta[idx_shared] * t_shared
        
        # likelihood 
        y_pred = pm.Normal("y_pred", mu = mu, sigma = eps, observed = y)
        
    return m0

## compile first model 
m0 = model_zero()

## model to graphviz (Kruschke). 
pm.model_to_graphviz(m0) 

### prior & posterior predictive checks ###
# https://docs.pymc.io/notebooks/posterior_predictive.html : same pattern for ppc & predictions
# https://docs.pymc.io/notebooks/multilevel_modeling.html : better structure?

'''
This workflow is by far the fastest if you have
a good idea about your priors & are unlikely
to want to change them. 
'''

### sample 
# convenience function
def sample_mod(
    model, 
    posterior_draws = 2000, 
    post_pred_draws = 1000,
    prior_pred_draws = 500):
    
    with model: 
        trace = pm.sample(return_inferencedata = False, draws = posterior_draws)
        post_pred = pm.sample_posterior_predictive(trace, samples = post_pred_draws)
        prior_pred = pm.sample_prior_predictive(samples = prior_pred_draws)
        m_idata = az.from_pymc3(trace = trace, posterior_predictive=post_pred, prior=prior_pred)
    
    return m_idata

m0_idata = sample_mod(m0)

### model checks
az.summary(m0_idata, round_to=2)

### prior & posterior predictions
# convenience function
def updating_check(m_idata, n_prior = 50, n_posterior = 50): 
    fig, axes = plt.subplots(nrows = 2)

    az.plot_ppc(m_idata, group = "prior", num_pp_samples = n_prior, ax = axes[0])
    az.plot_ppc(m_idata, num_pp_samples = n_posterior, ax = axes[1])
    plt.draw()
    
updating_check(m0_idata)

##### updating checks? checking the chain, etc? ######



## save data
m0_idata.to_netcdf("../models/m0_idata.nc")

########### m1 : random intercepts ##########
def model_one(): 
    
    with pm.Model() as m1: 
        
        # shared variables 
        t_shared = pm.Data('t_shared', t)
        idx_shared = pm.Data('idx_shared', idx)
        
        # hyper priors
        alpha_hyper = pm.Normal("alpha_hyper", mu = 0, sd = 0.5)
        
        # priors (group level?)
        alpha = pm.Normal("alpha", mu = alpha_hyper)

## load data
test = az.from_netcdf("../models/m0_idata.nc")
az.plot_ppc(test, num_pp_samples = 100)
