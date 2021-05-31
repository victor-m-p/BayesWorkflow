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
m_student_strict = fm.student(t_train, idx_train, y_train, n_train, 0.05)
m_student_reasonable = fm.student(t_train, idx_train, y_train, n_train, 0.5)
m_student_vague = fm.student(t_train, idx_train, y_train, n_train, 5)

## sample all the models ##
idata_student_strict = fh.sample_mod(m_student_strict) # fails spectacularly. 
idata_student_reasonable = fh.sample_mod(m_student_reasonable) # great
idata_student_vague = fh.sample_mod(m_student_vague) # great

## plot traces 
az.plot_trace(idata_student_strict)
az.plot_trace(idata_student_reasonable)
az.plot_trace(idata_student_vague)

## updating checks for all models 
fh.updating_check(idata_student_strict)
fh.updating_check(idata_student_reasonable)
fh.updating_check(idata_student_vague)

## save all the models 
idata_student_strict.to_netcdf("../models_python/idata_student_strict.nc")
idata_student_reasonable.to_netcdf("../models_python/idata_student_reasonable.nc")
idata_student_vague.to_netcdf("../models_python/idata_student_vague.nc")

