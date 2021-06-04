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

# fix plot trace
az.plot_trace(m_idata, figsize = (12, 12))
plt.savefig(f"../plots_python/multilevel_generic_plot_trace.jpeg")

# fix orange dots 
plt.scatter(
    t_train, 
    y_train, 
    color = "darkorange",
    alpha = 0.5)

# different HDI for parameters
fig, ax = plt.subplots(figsize = (10, 10))
az.plot_forest(
    m_idata,
    var_names=["alpha", "beta", "sigma"], 
    combined=True, # combine chains 
    kind='ridgeplot', # instead of default which does not show distribution
    ridgeplot_truncate=False, # do show the tails 
    hdi_prob = .8, # hdi prob .8 here. 
    ridgeplot_alpha = 0.5, # looks prettier
    ridgeplot_quantiles = [0.5], # show mean
    ax = ax # add to our axis
    )

fig.suptitle("Python/pyMC3: HDI intervals for parameters")
fig.tight_layout()
plt.plot()