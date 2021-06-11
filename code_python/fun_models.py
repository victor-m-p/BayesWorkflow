'''
19-5-21 (VMP).
functions for fitting the models
references:
1. structure: https://docs.pymc.io/notebooks/multilevel_modeling.html
'''

## packages
import pymc3 as pm
import numpy as np
import theano.tensor as tt

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

# actually random intercepts
def intercept(t, idx, y, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = dims)
        t_ = pm.Data('t_shared', t, dims = dims)

        # hyper-priors (group-level effects)
        alpha = pm.Normal("alpha", mu = 1.5, sigma = sigma)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma = sigma)
        
        # fixed slope for beta
        beta = pm.Normal("beta", mu = 0, sigma = sigma)
        
        # varying intercepts & slopes for alpha
        alpha_var = pm.Normal("alpha_var", mu = alpha, sigma = sigma_alpha, dims = "idx")
        
        # expected value per participant at each time-step
        mu = alpha_var[idx_] + beta * t_ 

        # model error
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y, dims = dims)

    return m

# actually multilevel
def covariation(t, idx, y, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = ('idx', 't'))
        t_ = pm.Data('t_shared', t, dims = ('idx', 't'))

        
        # prior stddev in intercepts & slopes (variation across counties):
        sd_dist = pm.HalfNormal.dist(sigma) # distribution. 

        # get back standard deviations and rho:
        ## eta = 1: uniform (higher --> more weight on low cor.)
        ## n = 2: number of predictors
        chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=1, sd_dist=sd_dist, compute_corr = True) 

        # priors for mean effects
        alpha = pm.Normal("alpha", mu = 1.5, sigma = sigma)
        beta = pm.Normal("beta", mu = 0, sigma = sigma)
        
        # population of varying effects
        alpha_beta = pm.MvNormal("alpha_beta", 
                                 mu = tt.stack([alpha, beta]), 
                                 chol = chol, 
                                 dims=("idx", "param"))
    
        # expected value per participant at each time-step
        mu = alpha_beta[idx_, 0] + alpha_beta[idx_, 1] * t_

        # model error
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y, dims = ('idx', 't'))
        
    return m

'''
# random intercepts and slopes
def multilevel(t, idx, y, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = dims)
        t_ = pm.Data('t_shared', t, dims = dims)

        # hyper-priors
        alpha_mu = pm.Normal("alpha_mu", mu = 1.5, sigma = sigma)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = sigma)
        beta_mu = pm.Normal("beta_mu", mu = 0, sigma = sigma)
        beta_sigma = pm.HalfNormal("beta_sigma", sigma = sigma)
        
        # varying intercepts & slopes
        alpha = pm.Normal("alpha", mu = alpha_mu, sigma = alpha_sigma, dims = "idx")
        beta = pm.Normal("beta", mu = beta_mu, sigma = beta_sigma, dims = "idx")
        
        # expected value per participant at each time-step
        mu = alpha[idx_] + beta[idx_] * t_

        # model error
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y, dims = dims)


    return m
'''

'''
# random intercepts and slopes (student-t)
def student(t, idx, y, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = dims)
        t_ = pm.Data('t_shared', t, dims = dims)

        # hyper-priors
        alpha_mu = pm.Normal("alpha_mu", mu = 1.5, sigma = sigma)
        alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = sigma)
        beta_mu = pm.Normal("beta_mu", mu = 0, sigma = sigma)
        beta_sigma = pm.HalfNormal("beta_sigma", sigma = sigma)
        
        # varying intercepts & slopes
        alpha = pm.Normal("alpha", mu = alpha_mu, sigma = alpha_sigma, dims = "idx")
        beta = pm.Normal("beta", mu = beta_mu, sigma = beta_sigma, dims = "idx")
        
        # expected value per participant at each time-step
        mu = alpha[idx_] + beta[idx_] * t_

        # nu
        v = pm.Gamma("v", alpha = 2, beta = 0.1)
        
        # model error
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # likelihood
        y_pred = pm.StudentT("y_pred", nu = v, mu = mu, sigma = sigma, observed = y, dims = dims)
        
    return m
'''