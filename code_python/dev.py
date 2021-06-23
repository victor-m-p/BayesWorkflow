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
RANDOM_SEED = 42

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

# more coords
coords["param"] = ["alpha", "beta"]
coords["param_bis"] = ["alpha", "beta"]

### compile the model (this cannot be saved, only the idata) ###
m = fm.covariation(
    t = t_train, 
    idx = idx_train, 
    y = y_train, 
    coords = coords, 
    dims = dims, 
    sigma = 0.5)

# load idata #
m_idata = az.from_netcdf("../models_python/idata_covariation_generic.nc")

### meta-data ###
model_type = "covariation"
prior_level = "generic"
kind = "individual"
ID = 0

for ID in idx_unique: 
    # only relevant idx
    ID_tmp = m_idata.posterior_predictive.sel(idx = ID)

    # small and large
    small = az.hdi(ID_tmp, hdi_prob = 0.8)["y_pred"]
    large = az.hdi(ID_tmp, hdi_prob = 0.95)["y_pred"]

    # y values for the right idx
    y = train[train["idx"] == ID].y.values

    # plot 
    fig, ax = plt.subplots(figsize = (10, 7)) 
    ax.scatter(
        t_unique, 
        y, 
        color = "darkorange",
        s = 50)
    ax.vlines(
        t_unique, 
        small.sel(hdi = "lower"), 
        small.sel(hdi = "higher"), 
        color = "orange", 
        alpha = 0.5,
        linewidth = 15)
    ax.vlines(
        t_unique, 
        large.sel(hdi = "lower"), 
        large.sel(hdi = "higher"), 
        color = "orange", 
        alpha = 0.3,
        linewidth = 15)
    plt.xticks(t_unique)
    fig.suptitle(f"Python/pyMC3: Prediction Intervals (Alien {ID})")
    fig.tight_layout()
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_HDI_{kind}_{ID}.jpeg",
                    dpi = 300)

