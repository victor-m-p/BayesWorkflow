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

## compile all the models 
m_pooled = fm.student(t_train, idx_train, y_train, n_train, 0.5)
m_multilevel = fm.student(t_train, idx_train, y_train, n_train, 0.5)
m_student = fm.student(t_train, idx_train, y_train, n_train, 0.5)

## load the idata 
idata_pooled = az.from_netcdf("../models_python/idata_pooled_reasonable.nc")
idata_multilevel = az.from_netcdf("../models_python/idata_multilevel_reasonable.nc")
idata_student  = az.from_netcdf("../models_python/idata_student_reasonable.nc")

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



