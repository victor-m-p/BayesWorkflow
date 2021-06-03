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
idx_train = train.idx.values.reshape((n_idx, n_time))

# gather dataset 
dataset = xr.Dataset(
    {'t_data': (dims, t_train),
    'y_data': (dims, y_train)},
    coords = coords)

#t_data = dataset['t_data']
#y_data = dataset['y_data']

with pm.Model(coords = coords) as model:
    
    # Inputs
    idx_ = pm.Data('idx_shared', idx_train, dims = dims)
    t_ = pm.Data('t_shared', t_train, dims = dims)

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
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = dims)

# sample posterior 
with model: 
    m_idata = pm.sample(
        draws = 2000, 
        tune = 2000, 
        random_seed = 32, 
        chains = 2,
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
y_pred_reshape = y_pred.reshape((4000*n_idx, n_time)) # 4000 = 2000 (draws) * 2 (chains)
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

ax.legend()
fig.suptitle("Python/pyMC3: prior predictive check")
fig.tight_layout()
figure = plt.gcf()
figure.set_size_inches(10, 7)
plt.savefig(f"../plots_python/test.jpeg",
            dpi = 300)
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

#### Predictions ####
with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

# get unique values for shared. 
t_unique_test = np.unique(test.t.values)
idx_unique_test = np.unique(test.idx.values)

# get n unique for shapes. 
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
    pm.set_data({"t_shared": t_test, "idx_shared": idx_test})
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