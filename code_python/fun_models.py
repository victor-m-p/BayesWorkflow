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
def pooled(t, y, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m:
        
        # shared variables
        t_ = pm.Data('t_shared', t, dims = dims)
        
        # specify priors for parameters & model error
        beta = pm.Normal("beta", mu = 0, sigma = sigma)
        alpha = pm.Normal("alpha", mu = 1.5, sigma = sigma)
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # calculate mu
        mu = alpha + beta * t_
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y)
        
    return m
        
# random intercepts and slopes
def multilevel(t, idx, y, n_id, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = dims)
        t_ = pm.Data('t_shared', t, dims = dims)

        # hyper-priors
        alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = 0.5)
        beta = pm.Normal("beta", mu = 0, sigma = 0.5)
        beta_sigma = pm.HalfNormal("beta_sigma", sigma = 0.5)
        
        # varying intercepts & slopes
        alpha_varying = pm.Normal("alpha_varying", mu = alpha, sigma = alpha_sigma, dims = "idx")
        beta_varying = pm.Normal("beta_varying", mu = beta, sigma = beta_sigma, dims = "idx")
        
        # expected value per participant at each time-step
        mu = alpha_varying[idx_] + beta_varying[idx_] * t_

        # model error
        sigma = pm.HalfNormal("sigma", sigma = 0.5)
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y, dims = dims)


    return m

# random intercepts and slopes (student-t)
def student(t, idx, y, n_id, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = dims)
        t_ = pm.Data('t_shared', t, dims = dims)

        # hyper-priors
        alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = 0.5)
        beta = pm.Normal("beta", mu = 0, sigma = 0.5)
        beta_sigma = pm.HalfNormal("beta_sigma", sigma = 0.5)
        
        # varying intercepts & slopes
        alpha_varying = pm.Normal("alpha_varying", mu = alpha, sigma = alpha_sigma, dims = "idx")
        beta_varying = pm.Normal("beta_varying", mu = beta, sigma = beta_sigma, dims = "idx")
        
        # expected value per participant at each time-step
        mu = alpha_varying[idx_] + beta_varying[idx_] * t_

        # nu
        v = pm.Gamma("v", alpha = 2, beta = 0.1)
        
        # model error
        sigma = pm.HalfNormal("sigma", sigma = 0.5)
        
        # likelihood
        y_pred = pm.StudentT("y_pred", nu = v, mu = mu, sigma = sigma, observed = y, dims = dims)
        
    return m
