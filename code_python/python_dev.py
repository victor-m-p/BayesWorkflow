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
y_train = train.y.values
idx_train, idx_unique = pd.factorize(train["idx"], sort=True)
n = len(train.idx) # obs id: unique.
t_unique = np.unique(t_train)

coords = {
    "obs_id": np.arange(n), # unique id.
    "idx": idx_unique, # unique idx. 
}


with pm.Model(coords = coords) as m: 
    
    # shared variables
    t_shared = pm.Data('t_shared', t_train, dims = "obs_id")
    idx_shared = pm.Data('idx_shared', idx_train, dims = "obs_id")
    
    # hyper-priors
    alpha_hyper = pm.Normal("alpha_hyper", mu = 1.5, sigma = 0.5)
    alpha_sigma_hyper = pm.HalfNormal("alpha_sigma_hyper", sigma = 0.5)
    beta_hyper = pm.Normal("beta_hyper", mu = 0, sigma = 0.5)
    beta_sigma_hyper = pm.HalfNormal("beta_sigma_hyper", sigma = 0.5)
    
    # varying intercepts & slopes
    alpha = pm.Normal("alpha", mu = alpha_hyper, sigma = alpha_sigma_hyper, dims = "idx")
    beta = pm.Normal("beta", mu = beta_hyper, sigma = beta_sigma_hyper, dims = "idx")
    
    # expected value per participant at each time-step
    mu = alpha[idx_shared] + beta[idx_shared] * t_shared
    
    # model error
    sigma = pm.HalfNormal("sigma", sigma = 0.5)
    
    # store fitted 
    # fitted = pm.Normal("fitted", mu = alpha + beta * t_shared, sigma)
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = "obs_id")

with m: 
    m_idata = pm.sample(
        2000, 
        tune=2000, 
        target_accept=0.99, 
        random_seed=32, 
        return_inferencedata=True
    )

# try posterior predictive
with m:
    post_pred = pm.sample_posterior_predictive(
        m_idata,
        var_names = ["y_pred", "alpha", "beta"])
    idata_aux = az.from_pymc3(posterior_predictive=post_pred)

m_idata.extend(idata_aux)

# check it out: these look ok. 
m_idata.posterior_predictive["alpha"].shape
m_idata.posterior_predictive["y_pred"].shape


m_idata.posterior_predictive["y_pred"].mean(dim=("chain", "draw"))


# check the constant data
m_idata.constant_data
t_shared = m_idata.constant_data.t_shared
idx_shared = m_idata.constant_data.idx_shared
post_pred = m_idata.posterior_predictive.assign_coords(t_shared=t_shared, idx_shared=idx_shared)
post_pred
mean_postpred = post_pred["y_pred"].mean(dim = ("chain", "draw")).sortby("t_shared", "idx_shared")

## 


#### second model ####
coords2 = {
    "obs_id": np.arange(n), # unique id.
    "idx": idx_unique, # unique idx. 
    "t_unique": t_unique
}

with pm.Model(coords = coords2) as m2: 
    
    # shared variables
    t_shared = pm.Data('t_shared', t_train, dims = "obs_id")
    idx_shared = pm.Data('idx_shared', idx_train, dims = "obs_id")
    
    # hyper-priors
    alpha_hyper = pm.Normal("alpha_hyper", mu = 1.5, sigma = 0.5)
    alpha_sigma_hyper = pm.HalfNormal("alpha_sigma_hyper", sigma = 0.5)
    beta_hyper = pm.Normal("beta_hyper", mu = 0, sigma = 0.5)
    beta_sigma_hyper = pm.HalfNormal("beta_sigma_hyper", sigma = 0.5)
    
    # varying intercepts & slopes
    alpha = pm.Normal("alpha", mu = alpha_hyper, sigma = alpha_sigma_hyper, dims = "idx")
    beta = pm.Normal("beta", mu = beta_hyper, sigma = beta_sigma_hyper, dims = "idx")
    
    # expected value per participant at each time-step
    mu = alpha[idx_shared] + beta[idx_shared] * t_shared
    
    # model error
    sigma = pm.HalfNormal("sigma", sigma = 0.5)
    
    # store fitted 
    # fitted = pm.Normal("fitted", mu = alpha + beta * t_shared, sigma)
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = "obs_id")


with m2: 
    m_idata2 = pm.sample(
        2000, 
        tune=2000, 
        target_accept=0.99, 
        random_seed=32, 
        return_inferencedata=True
    )

# try posterior predictive
with m2:
    post_pred2 = pm.sample_posterior_predictive(
        m_idata2,
        var_names = ["y_pred", "alpha", "beta"])
    idata_aux2 = az.from_pymc3(posterior_predictive=post_pred2)
    
m_idata2.extend(idata_aux2)
ppc = m_idata.posterior_predictive
ppc
## mu pp. 
mu_pp = (ppc["alpha_hyper"] + ppc["beta_hyper"] * t_unique[:, None]).T
t_unique

### plotting some stuff
fig, ax = plt.subplots()
ax.plot(t_train, y_train, 'o')

### get average actor (only fixed). 
## following burkner. 

