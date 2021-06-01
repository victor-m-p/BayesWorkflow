## packages ##
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt 
import pymc3 as pm
import seaborn as sns
import theano 
import arviz as az
import pickle
import fun_models as fm
import fun_helper as fh
import dataframe_image as dfi

### load data ###
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)
    
# take out the vectors
t_train = train.t.values
idx_train = train.idx.values
y_train = train.y.values
n_train = len(np.unique(idx_train))

with pm.Model() as m: 
    
    # shared variables
    t_shared = pm.Data('t_shared', train.t.values)
    idx_shared = pm.Data('idx_shared', idx)
    
    # hyper-priors
    alpha_hyper = pm.Normal("alpha_hyper", mu = 1.5, sigma = sigma)
    alpha_sigma_hyper = pm.HalfNormal("alpha_sigma_hyper", sigma = sigma)
    beta_hyper = pm.Normal("beta_hyper", mu = 0, sigma = sigma)
    beta_sigma_hyper = pm.HalfNormal("beta_sigma_hyper", sigma = sigma)
    
    # varying intercepts & slopes
    alpha = pm.Normal("alpha", mu = alpha_hyper, sigma = alpha_sigma_hyper, shape = n_id)
    beta = pm.Normal("beta", mu = beta_hyper, sigma = beta_sigma_hyper, shape = n_id)
    
    # expected value per participant at each time-step
    mu = alpha[idx_shared] + beta[idx_shared] * t_shared
    
    # model error
    sigma = pm.HalfNormal("sigma", sigma = sigma)
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y)
