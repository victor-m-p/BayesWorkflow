# https://discourse.pymc.io/t/creating-hierarchical-models-using-3-dimensional-xarray-data/5900/8

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
import xarray as xr

### load data ###
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)

# get unique stuff.  
t_unique = np.unique(train.t.values)
idx_unique = np.unique(train.idx.values)

# get n of unique for shapes
n_time = len(t_unique)
n_idx = len(idx_unique)

# create coords and dims 
coords = {
    'idx': idx_unique,
    't': t_unique
}

dims = coords.keys()

# data in correct format. 
t_train = train.t.values.reshape((n_idx, n_time))
y_train = train.y.values.reshape((n_idx, n_time))

# gather dataset 
dataset = xr.Dataset(
    {'t_data': (dims, t_train),
    'y_data': (dims, y_train)},
    coords = coords)

with pm.Model(coords = coords) as model:
    
    # Inputs
    t_ = pm.Data('t_shared', t_train, dims = dims)
    
    # priors
    alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
    beta = pm.Normal("beta", mu = 0, sigma = 0.5)
    
    # expected value per participant at each time-step
    mu = alpha + beta * t_

    # model error
    sigma = pm.HalfNormal("sigma", sigma = 0.5)
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = dims)

# sample posterior 
with model: 
    m_idata = pm.sample(
        draws = 2000, 
        tune=2000, 
        random_seed=32, 
        return_inferencedata=True
    )

# sample posterior predictive
with model:
    post_pred = pm.sample_posterior_predictive(
        m_idata,
        var_names = [
            "y_pred", 
            "alpha",
            "beta"])
    idata_aux = az.from_pymc3(posterior_predictive=post_pred)
    
m_idata.extend(idata_aux)

# take out posterior predictive 
ppc = m_idata.posterior_predictive

##### plot full uncertainty #####
## checks out against brms ##

# preprocessing 
y_pred = ppc["y_pred"].mean(axis = 0).values
y_pred_reshape = y_pred.reshape((4000*n_idx, n_time))
y_mean = y_pred.mean(axis = (0, 1))

# plot it with the data. 
fig, ax = plt.subplots()

# plot data. 
ax.plot(t_train.flatten(), y_train.flatten(), 'o')

# plot mean
ax.plot(t_unique, y_mean)

# plot 94% HDI. 
az.plot_hdi(
    t_unique,
    y_pred_reshape,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": "Prediction intervals 80% HPD"},
    hdi_prob = 0.8
)

az.plot_hdi(
    t_unique,
    y_pred_reshape,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": "Prediction intervals 95% HDI"},
    hdi_prob = 0.95
)

# we need title & legend. 

#### fixed approach #### 
y_fixed = (ppc.alpha.values + ppc.beta.values * t_unique[:, None]).T

# plot it with the data
fig, ax = plt.subplots(figsize = (10, 7))

# plot data. 
ax.plot(t_train.flatten(), y_train.flatten(), 'o')

# plot mean
ax.plot(t_unique, y_mean)

# plot 94% HDI. 
az.plot_hdi(
    t_unique,
    y_fixed,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": "Prediction intervals 80% HPD"},
    hdi_prob = 0.8
)

az.plot_hdi(
    t_unique,
    y_fixed,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": "Prediction intervals 95% HDI"},
    hdi_prob = 0.95
)

