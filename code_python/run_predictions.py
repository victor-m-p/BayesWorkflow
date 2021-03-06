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

### Predictions ###
# load test data
test = pd.read_csv("../data/test.csv")

# get unique values for shared. 
t_unique_test = np.unique(test.t.values)
idx_unique_test = np.unique(test.idx.values)

# get n unique for shapes. 
n_time_test = len(t_unique_test)
n_idx_test = len(idx_unique_test)

# new coords as well
prediction_coords = {
    'idx': idx_unique_test,
    't': t_unique_test
}

# test data in correct format. 
t_test = test.t.values.reshape((n_idx_test, n_time_test))
y_test = test.y.values.reshape((n_idx_test, n_time_test))
idx_test = test.idx.values.reshape((n_idx_test, n_time_test))

with m:
    pm.set_data({"t_shared": t_test, "idx_shared": idx_test})
    stl_pred = pm.fast_sample_posterior_predictive(
        m_idata.posterior, random_seed=RANDOM_SEED
    )
    az.from_pymc3_predictions(
        stl_pred, idata_orig=m_idata, inplace=True, coords=prediction_coords
    )

# plot hdi for prediction
fh.plot_hdi(
    t = t_test,
    y = y_test,
    n_idx = n_idx_test,
    m_idata = m_idata,
    model_type = "covariation",
    prior_level = "generic",
    kind = "predictions"
)

model_type = "covariation"
prior_level = "generic"

# plot hdi for individual aliens
for ID in idx_unique_test: 
    # only relevant idx
    ID_tmp = m_idata.predictions.sel(idx = ID)

    # small and large hdi interval
    hdi1 = az.hdi(ID_tmp, hdi_prob = 0.8)["y_pred"]
    hdi2 = az.hdi(ID_tmp, hdi_prob = 0.95)["y_pred"]

    # y values for the right idx
    y = test[test["idx"] == ID].y.values
    
    fh.hdi_ID(
        t_unique = t_unique_test, 
        y = y, 
        hdi1 = hdi1, 
        hdi2 = hdi2, 
        model_type = model_type, 
        prior_level = prior_level,
        type = "test",
        ID = ID)