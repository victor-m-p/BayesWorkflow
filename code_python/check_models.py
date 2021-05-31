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
m_rand_int_slope = fm.rand_int_slope(t_train, idx_train, y_train, n_train)
m_student = fm.student(t_train, idx_train, y_train, n_train)

## load the idata ## 
idata_pooled = az.from_netcdf("../models/idata_pooled.nc")
idata_rand_int_slope = az.from_netcdf("../models/idata_rand_int_slope.nc")
idata_student = az.from_netcdf("../models/idata_student.nc")

### Kruschke plots (graphviz) ###
# pm.model_to_graphviz(m0) 

## plot trace (plot forest?).
az.plot_trace(idata_pooled)
az.plot_trace(idata_rand_int_slope)
az.plot_trace(idata_student)

## model comparison
loo_overview = az.compare({
    "m_pooled": idata_pooled,
    "m_rand_int": idata_rand_int,
    "m_rand_int_slope": idata_rand_int_slope,
    "m_student": idata_student})

## 