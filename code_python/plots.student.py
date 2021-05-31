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

## compile all the models ##
m_student_strict = fm.student(t_train, idx_train, y_train, n_train, 0.05)
m_student_reasonable = fm.student(t_train, idx_train, y_train, n_train, 0.5)
m_student_vague = fm.student(t_train, idx_train, y_train, n_train, 5)

## load the idata ## 
idata_student_strict = az.from_netcdf("../models_python/idata_student_strict.nc")
idata_student_reasonable = az.from_netcdf("../models_python/idata_student_reasonable.nc")
idata_student_vague = az.from_netcdf("../models_python/idata_student_vague.nc")

# plot plate 
fh.plot_plate(
    compiled_model = m_student_reasonable,
    model_type = "student"
)

## plot prior predictive & save.
fh.prior_pred(
    m_idata = idata_student_strict, 
    model_type = "student",
    prior_level = "strict",
    n_draws = 100
)

fh.prior_pred(
    m_idata = idata_student_reasonable,
    model_type = "student",
    prior_level = "reasonable",
    n_draws = 100
)

fh.prior_pred(
    m_idata = idata_student_vague,
    model_type = "student",
    prior_level = "vague",
    n_draws = 100
)

## posterior predictive & save
fh.posterior_pred(
    m_idata = idata_student_strict,
    model_type = "student",
    prior_level = "strict",
    n_draws = 100
)

fh.posterior_pred(
    m_idata = idata_student_reasonable,
    model_type = "student",
    prior_level = "reasonable",
    n_draws = 100
)

fh.posterior_pred(
    m_idata = idata_student_vague,
    model_type = "student",
    prior_level = "vague",
    n_draws = 100
)

## Plot traces
fh.plot_trace(
    m_idata = idata_student_strict,
    model_type = "student",
    prior_level = "strict"
)

fh.plot_trace(
    m_idata = idata_student_reasonable,
    model_type = "student",
    prior_level = "reasonable"
)

fh.plot_trace(
    m_idata = idata_student_vague,
    model_type = "student",
    prior_level = "vague"
)

## Plot summary
fh.export_summary(
    m_idata = idata_student_strict,
    model_type = "student",
    prior_level = "strict"
)

fh.export_summary(
    m_idata = idata_student_reasonable,
    model_type = "student",
    prior_level = "reasonable"
)

fh.export_summary(
    m_idata = idata_student_vague,
    model_type = "student",
    prior_level = "vague"
)