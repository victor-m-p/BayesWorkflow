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

### load data ###
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

# take out the vectors
t_train = train.t.values
idx_train = train.idx.values
y_train = train.y.values
n_train = len(np.unique(idx_train))

## compile all the models ##
m_pooled_strict = fm.pooled(t_train, idx_train, y_train, 0.05)
m_pooled_reasonable = fm.pooled(t_train, idx_train, y_train, 0.5)
m_pooled_vague = fm.pooled(t_train, idx_train, y_train, 5)

## sample for all the models
idata_pooled_strict = fh.sample_mod(m_pooled_strict)
idata_pooled_reasonable = fh.sample_mod(m_pooled_reasonable) 
idata_pooled_vague = fh.sample_mod(m_pooled_vague)

## plot traces 
az.plot_trace(idata_pooled_strict)
az.plot_trace(idata_pooled_reasonable)
az.plot_trace(idata_pooled_vague)

## updating checks for all models 
fh.updating_check(idata_pooled_strict)
fh.updating_check(idata_pooled_reasonable)
fh.updating_check(idata_pooled_vague)

## save all the models 
idata_pooled_strict.to_netcdf("../models_python/idata_pooled_strict.nc")
idata_pooled_reasonable.to_netcdf("../models_python/idata_pooled_reasonable.nc")
idata_pooled_vague.to_netcdf("../models_python/idata_pooled_vague.nc")

