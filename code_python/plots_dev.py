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

## try to make a plot of just the pooled
test = idata_pooled.posterior_predictive["y_pred"].values
test.shape
test2 = idata_multilevel.posterior_predictive["y_pred"].values
test2.shape
idata_student.posterior_predictive
_, ax = plt.subplots()
ax.plot(t_train, y_train, "o", ms=4, alpha=0.4, label="Data")
az.plot_hdi(
    t_train,
    idata_pooled.posterior_predictive["y_pred"],
    fill_kwargs={"alpha": 0.8, "color": "#a1dab4", "label": "Outcome 94% HPD"},
)

# testing multilevel 
idata_multilevel.posterior

az.plot_hdi(
    t_train,
    idata_multilevel.posterior_predictive["y_pred"],
    fill_kwargs={"alpha": 0.8, "color": "#a1dab4", "label": "Outcome 94% HPD"}
)

?az.plot_hdi

az.plot_hdi(
    t_train,
    idata_student.posterior_predictive["y_pred"],
    fill_kwargs={"alpha": 0.8, "color": "#a1dab4", "label": "Outcome 94% HPD"}
)

test = idata_multilevel.posterior_predictive["y_pred"].values
test
#https://docs.pymc.io/notebooks/posterior_predictive.html
mu_pp = (ppc["a"] + ppc["b"] * predictor_scaled[:, None]).T

_, ax = plt.subplots()

ax.plot(predictor_scaled, outcome_scaled, "o", ms=4, alpha=0.4, label="Data")
ax.plot(predictor_scaled, mu_pp.mean(0), label="Mean outcome", alpha=0.6)
az.plot_hpd(
    predictor_scaled,
    mu_pp,
    ax=ax,
    fill_kwargs={"alpha": 0.8, "label": "Mean outcome 94% HPD"},
)
az.plot_hpd(
    predictor_scaled,
    ppc["obs"],
    ax=ax,
    fill_kwargs={"alpha": 0.8, "color": "#a1dab4", "label": "Outcome 94% HPD"},
)

ax.set_xlabel("Predictor (stdz)")
ax.set_ylabel("Outcome (stdz)")
ax.set_title("Posterior predictive checks")
ax.legend(ncol=2, fontsize=10);

