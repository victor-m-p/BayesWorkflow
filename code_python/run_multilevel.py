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
m_multilevel_restrictive = fm.multilevel(t_train, idx_train, y_train, n_train, 0.05)
m_multilevel_reasonable = fm.multilevel(t_train, idx_train, y_train, n_train, 0.5)
m_multilevel_vague = fm.multilevel(t_train, idx_train, y_train, n_train, 5)

## sample all the models ##
idata_multilevel_restrictive = fh.sample_mod(m_multilevel_restrictive)
idata_multilevel_reasonable = fh.sample_mod(m_multilevel_reasonable) 
idata_multilevel_vague = fh.sample_mod(m_multilevel_vague)

## plot traces 
az.plot_trace(idata_multilevel_restrictive) # "The number of effective samples is smaller than 25% for some parameters" 
az.plot_trace(idata_multilevel_reasonable)
az.plot_trace(idata_multilevel_vague)

## updating checks for all models 
fh.updating_check(idata_multilevel_restrictive)
fh.updating_check(idata_multilevel_reasonable)
fh.updating_check(idata_multilevel_vague)

## save all the models 
idata_multilevel_restrictive.to_netcdf("../models_python/idata_multilevel_restrictive.nc")
idata_multilevel_reasonable.to_netcdf("../models_python/idata_multilevel_reasonable.nc")
idata_multilevel_vague.to_netcdf("../models_python/idata_multilevel_vague.nc")

