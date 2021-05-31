'''
19-5-21 (VMP).
functions for fitting the models
references:
1. structure: https://docs.pymc.io/notebooks/multilevel_modeling.html
'''

## packages
import pymc3 as pm
import numpy as np


# pooled
def pooled(t, idx, y, sigma): 
    
    with pm.Model() as m:
        
        # shared variables
        t_shared = pm.Data('t_shared', t)
        
        # specify priors for parameters & model error
        beta = pm.Normal("beta", mu = 0, sigma = sigma)
        alpha = pm.Normal("alpha", mu = 1.5, sigma = sigma)
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # calculate mu
        mu = alpha + beta * t_shared
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y)
        
    return m
        
# random intercepts and slopes
def multilevel(t, idx, y, n_id, sigma): 
    
    with pm.Model() as m: 
        
        # shared variables
        t_shared = pm.Data('t_shared', t)
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

    return m

# random intercepts and slopes (student-t)
def student(t, idx, y, n_id, sigma): 
    
    with pm.Model() as m: 
        
        # shared variables
        t_shared = pm.Data('t_shared', t)
        idx_shared = pm.Data('idx_shared', idx)
        
        # hyper-priors
        alpha_hyper = pm.Normal("alpha_hyper", mu = 1.5, sigma = sigma)
        alpha_sigma_hyper = pm.HalfNormal("alpha_sigma_hyper", sigma = sigma)
        beta_hyper = pm.Normal("beta_hyper", mu = 0, sigma = sigma)
        beta_sigma_hyper = pm.HalfNormal("beta_sigma_hyper", sigma = sigma)
        
        # varying intercepts & slopes
        alpha = pm.Normal("alpha", mu = alpha_hyper, sigma = alpha_sigma_hyper, shape = n_id)
        beta = pm.Normal("beta", mu = beta_hyper, sigma = beta_sigma_hyper, shape = n_id)
        v = pm.Gamma("v", alpha = 2, beta = 0.1)
        
        # expected value per participant at each time-step
        mu = alpha[idx_shared] + beta[idx_shared] * t_shared
        
        # model error
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # likelihood
        y_pred = pm.StudentT("y_pred", nu = v, mu = mu, sigma = sigma, observed = y)
    
    return m
