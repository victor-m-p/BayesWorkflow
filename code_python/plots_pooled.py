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

## load the idata ## 
idata_pooled_strict = az.from_netcdf("../models_python/idata_pooled_strict.nc")
idata_pooled_reasonable = az.from_netcdf("../models_python/idata_pooled_reasonable.nc")
idata_pooled_vague = az.from_netcdf("../models_python/idata_pooled_vague.nc")

'''
## kruschke diagram
pm.model_to_graphviz(m_pooled_reasonable) # has to be rendered real-time.
'''

## plot prior predictive & save.
fh.prior_pred(
    m_idata = idata_pooled_strict, 
    model_type = "pooled",
    prior_level = "strict",
    n_draws = 100
)

fh.prior_pred(
    m_idata = idata_pooled_reasonable,
    model_type = "pooled",
    prior_level = "reasonable",
    n_draws = 100
)

fh.prior_pred(
    m_idata = idata_pooled_vague,
    model_type = "pooled",
    prior_level = "vague",
    n_draws = 100
)

## posterior predictive & save
fh.posterior_pred(
    m_idata = idata_pooled_strict,
    model_type = "pooled",
    prior_level = "strict",
    n_draws = 100
)

fh.posterior_pred(
    m_idata = idata_pooled_reasonable,
    model_type = "pooled",
    prior_level = "reasonable",
    n_draws = 100
)

fh.posterior_pred(
    m_idata = idata_pooled_vague,
    model_type = "pooled",
    prior_level = "vague",
    n_draws = 100
)

'''
## Plot traces

In PyMC3 I additionally like to plot the traces

'''

'''
## plot traces 
az.plot_trace(idata_pooled_strict)
figure = plt.gcf()
#figure.set_size_inches(10, 7)
plt.savefig("../plots_python/pooled_strict_trace.jpeg", dpi=300, transparent=False,
            facecolor=figure.get_facecolor(), edgecolor='none')

az.plot_trace(idata_pooled_reasonable)
az.plot_trace(idata_pooled_vague)

'''