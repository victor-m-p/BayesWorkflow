#import streamlit as st
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

## load data ##
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

## take out the arrays ##
t_train = train.t.values
idx_train = train.idx.values
y_train = train.y.values
n_train = len(np.unique(idx_train))

## compile all the models ##
m_pooled = fm.pooled(t_train, idx_train, y_train)
m_rand_int_slope = fm.multilevel(t_train, idx_train, y_train, n_train)
m_student = fm.student(t_train, idx_train, y_train, n_train)

## load the idata ## 
idata_unpooled = az.from_netcdf("../models/idata_unpooled.nc")
idata_pooled = az.from_netcdf("../models/idata_pooled.nc")
idata_rand_int = az.from_netcdf("../models/idata_rand_int.nc")
idata_rand_int_slope = az.from_netcdf("../models/idata_rand_int_slope.nc")
idata_student = az.from_netcdf("../models/idata_student.nc")

## take out the arrays ##
t_test = test.t.values
idx_test = test.idx.values
y_test = test.y.values

## 
idata_rand_int_slope


## make this into a function ## 
# https://docs.pymc.io/notebooks/multilevel_modeling.html
with m_rand_int_slope: 
    pm.set_data({
        "t_shared": t_test,
        "idx_shared": idx_test})
    predictions = pm.sample_posterior_predictive(idata_rand_int_slope)
    az.from_pymc3_predictions(
        predictions, idata_orig=idata_rand_int_slope, inplace=True
    )

## save it 
idata_rand_int_slope
## take out coords ## 
t_shared = idata_rand_int_slope.predictions_constant_data.t_shared
idx_shared = idata_rand_int_slope.predictions_constant_data.idx_shared

## assign coords ## 
predictions = idata_rand_int_slope.predictions.assign_coords(t_shared = t_shared, idx_shared = idx_shared)
predictions


## mean --> need to make this work ## 
avg = predictions["y_pred"].mean(dim=("chain", "draw", "idx_shared")).sortby("t_shared")




idata_rand_int_slope.predictions_constant_data.t_shared
idata_rand_int_slope.predictions
az.plot_hdi(idata_rand_int_slope.predictions_constant_data.t_shared,
            idata_rand_int_slope.predictions["y_pred"])

idata_rand_int_slope.predictions["y_pred"].mean(axis = 1)