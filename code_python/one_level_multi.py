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

import xarray as xr
import arviz as az
import pymc3 as pm
import numpy as np

######### own stuff ###########

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

idx_train = train.idx.values.reshape((n_idx, n_time))
t_data = dataset['t_data']
y_data = dataset['y_data']

with pm.Model(coords = coords) as model:
    
    # Inputs
    idx_shared = pm.Data('idx_', idx_train, dims = dims)
    t_shared = pm.Data('t_', t_data, dims = dims)

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
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = dims)

# 
with model: 
    m_idata = pm.sample(
        2000, 
        tune=2000, 
        random_seed=32, 
        return_inferencedata=True
    )

# try posterior predictive
with model:
    post_pred = pm.sample_posterior_predictive(
        m_idata,
        var_names = [
            "y_pred", 
            "alpha", 
            "beta",
            "alpha_hyper",
            "alpha_sigma_hyper",
            "beta_hyper",
            "beta_sigma_hyper"])
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
y_fixed = (ppc.alpha_hyper.values + ppc.beta_hyper.values * t_unique[:, None]).T

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

#### Predictions ####
with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

# get unique stuff.  
t_unique_test = np.unique(test.t.values)
idx_unique_test = np.unique(test.idx.values)

# get n of unique for shapes
n_time_test = len(t_unique_test)
n_idx_test = len(idx_unique_test)

# test data in correct format. 
t_test = test.t.values.reshape((n_idx_test, n_time_test))
y_test = test.y.values.reshape((n_idx_test, n_time_test))
idx_test = test.idx.values.reshape((n_idx_test, n_time_test))

# new coords as well
prediction_coords = {
    'idx': idx_unique_test,
    't': t_unique_test
}

with model:
    pm.set_data({"t_": t_test, "idx_": idx_test})
    stl_pred = pm.fast_sample_posterior_predictive(
        m_idata.posterior, random_seed=32
    )
    az.from_pymc3_predictions(
        stl_pred, idata_orig=m_idata, inplace=True, coords=prediction_coords
    )

ppc_unseen = m_idata.predictions
y_pred = ppc_unseen["y_pred"].mean(axis = 0).values
y_pred_reshape = y_pred.reshape((4000*n_idx_test, n_time_test))
y_mean = y_pred.mean(axis = (0, 1))

# plot it with the data
fig, ax = plt.subplots(figsize = (10, 7))

# plot data. 
ax.plot(t_test.flatten(), y_test.flatten(), 'o')

# plot mean
ax.plot(t_unique_test, y_mean)

# plot 94% HDI. 
az.plot_hdi(
    t_unique_test,
    y_pred_reshape,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": "Prediction intervals 80% HPD"},
    hdi_prob = 0.8
)

az.plot_hdi(
    t_unique_test,
    y_pred_reshape,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": "Prediction intervals 95% HDI"},
    hdi_prob = 0.95
)