## packages ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pymc3 as pm
import seaborn as sns
import arviz as az
import fun_models as fm
import fun_helper as fh
import dataframe_image as dfi
import xarray as xr

### load data ###
train = pd.read_csv("../data/train.csv")

### preprocessing ###
# t & idx unique
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

# take out dims 
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


## troubleshooting error for pooled
m = fm.pooled(
    t = t_train,
    y = y_train,
    coords = coords,
    dims = dims,
    sigma = 0.5
)

# idata
m_idata = az.from_netcdf("../models_python/idata_pooled_specific.nc")
posterior = m_idata.posterior_predictive
posterior.alpha.values.shape

outcome = (posterior.alpha.values + posterior.beta.values * t_unique[:, None]).T
outcome.shape
y_mean = outcome.mean(axis = 0)

# set-up plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange",
    alpha = 0.5)

# plot mean
ax.plot(t_unique, y_mean,
        color = "darkorange")

low = 0.8
# plot lower interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": f"{low*100}% HPD intervals"},
    hdi_prob = low)


# same schema for multilevel


### compile the model (this cannot be saved, only the idata) ###
m = fm.multilevel(
    t = t_train, 
    idx = idx_train, 
    y = y_train, 
    coords = coords, 
    dims = dims, 
    sigma = 0.5)

# idata
m_idata = az.from_netcdf("../models_python/idata_multilevel_generic.nc")
m_idata = az.from_netcdf("../models_python/idata_multilevel_specific.nc")

# try with alpha_varying & beta_varying

# try with alpha and beta
high, low = (.95, .8)
t_unique = np.unique(t_train)
n_time = len(t_unique)
posterior = m_idata.posterior
alpha = posterior.alpha_mu.values.reshape((1, 4000))
beta = posterior.beta_mu.values.reshape((1, 4000))
#alpha = posterior["alpha"].values.reshape((1, 4000))
#alpha = posterior.alpha.mean(axis = 2).values.reshape((1, 4000))
#beta = posterior["beta"].values.reshape((1, 4000))
#beta = posterior.beta.mean(axis = 2).values.reshape((1, 4000))
outcome = (alpha + beta * t_unique[:, None]).T
y_mean = outcome.mean(axis = 0)

# 
# set-up plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange",
    alpha = 0.5)

# plot mean
ax.plot(t_unique, y_mean,
        color = "darkorange")

# plot lower interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": f"{low*100}% HPD intervals"},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": f"{high*100}% HDI intervals"},
    hdi_prob = high)





### mean over random effects?
ppc = m_idata.posterior_predictive

high, low = (.95, .8)
t_unique = np.unique(t_train)
n_time = len(t_unique)

# take out data from ppc.
y_pred = ppc["y_pred"].mean(axis = 0).values

# mean over random effects and global mean
mean_rand = y_pred.mean(axis = 1)
y_mean = y_pred.mean(axis = (0, 1))

#outcome = (ppc.alpha.values + ppc.beta.values * t_unique[:, None]).T
#y_mean = outcome.mean(axis = 0)

# set-up plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange",
    alpha = 0.5)

# plot mean
ax.plot(t_unique, y_mean,
        color = "darkorange")

# plot lower interval
az.plot_hdi(
    t_unique,
    mean_rand,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": f"{low*100}% HPD intervals"},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    mean_rand,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": f"{high*100}% HDI intervals"},
    hdi_prob = high)