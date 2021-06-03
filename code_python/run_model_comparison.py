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

### compile all the (generic) models ###
m_pooled = fm.pooled(
    t = t_train,
    y = y_train,
    coords = coords,
    dims = dims,
    sigma = 0.5
)

m_multilevel = fm.multilevel(
    t = t_train, 
    idx = idx_train, 
    y = y_train, 
    coords = coords, 
    dims = dims, 
    sigma = 0.5)

m_student = fm.student(
    t = t_train, 
    idx = idx_train, 
    y = y_train, 
    coords = coords, 
    dims = dims, 
    sigma = 0.5
)

## load the idata 
idata_pooled = az.from_netcdf("../models_python/idata_pooled_generic.nc")
idata_multilevel = az.from_netcdf("../models_python/idata_multilevel_generic.nc")
idata_student  = az.from_netcdf("../models_python/idata_student_generic.nc")

## model comparison
loo_overview = az.compare({
    "m_pooled": idata_pooled,
    "m_multilevel": idata_multilevel,
    "m_student": idata_student})

## export it 
dfi.export(
    obj = loo_overview,
    filename = "../plots_python/loo_comparison.png"
)

# intervals: https://docs.pymc.io/notebooks/posterior_predictive.html



