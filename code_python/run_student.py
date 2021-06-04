## packages ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pymc3 as pm
import seaborn as sns
import theano 
import arviz as az
import pickle
import fun_models as fm
import fun_helper as fh
import xarray as xr

# set some global parameters
model_type = "student" # only thing different from run_multilevel.py
n_pp = 100
n_draws = 2000 
n_tune = 2000
n_chains = 2
target_accept = .99 ### CHANGE 
max_treedepth = 20 ### CHANGE 
prior_draws = 500
RANDOM_SEED = 42

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
idx_train = train.idx.values.reshape((n_idx, n_time)) # what differs from run_pooled.py

# gather dataset 
dataset = xr.Dataset(
    {'t_data': (dims, t_train),
    'y_data': (dims, y_train)},
    coords = coords)

## run main analysis for all multilevel model ##
for sigma, prior_level in [(0.05, "specific"), (0.5, "generic"), (5, "weak")]:
    
    # compile the model 
    m = fm.student(
        t = t_train, 
        idx = idx_train, 
        y = y_train, 
        coords = coords, 
        dims = dims, 
        sigma = sigma)
    
    # plot plate
    fh.plot_plate(
    compiled_model = m,
    model_type = model_type)
    
    # sample prior predictive
    # https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html
    with m:
        prior = pm.sample_prior_predictive(700) # small, but only for pp. 
        m_idata = az.from_pymc3(prior=prior)
    
    # prior predictive
    fh.prior_pred(
        m_idata = m_idata, 
        model_type = model_type,
        prior_level = prior_level,
        n_draws = n_pp)
    
    # sample posterior (add optimization). 
    with m:
        idata_posterior = pm.sample(
            draws = n_draws, 
            tune = n_tune, 
            chains = n_chains,
            return_inferencedata = True,
            target_accept = target_accept,
            max_treedepth = max_treedepth,
            random_seed = RANDOM_SEED)
    m_idata.extend(idata_posterior) # we can extend our existing idata.
    
    # check trace
    fh.plot_trace(
        m_idata = m_idata,
        model_type = model_type,
        prior_level = prior_level)
    
    # plot summary
    fh.export_summary(
        m_idata = m_idata,
        model_type = model_type,
        prior_level = prior_level)
    
    # sample posterior predictive
    with m:
        post_pred = pm.sample_posterior_predictive(
            m_idata,
            var_names = [
                "y_pred",
                "alpha",
                "beta"])
        idata_postpred = az.from_pymc3(posterior_predictive=post_pred)
    m_idata.extend(idata_postpred)

    # posterior predictive 
    fh.posterior_pred(
        m_idata = m_idata,
        model_type = model_type,
        prior_level = prior_level,
        n_draws = n_pp)
    
    # plot hdi for fixed effects
    fh.plot_hdi(
        t = t_train,
        y = y_train,
        n_idx = n_idx,
        m_idata = m_idata,
        model_type = model_type,
        prior_level = prior_level,
        kind = "fixed"
    )
    
    # plot hdi for all effects
    fh.plot_hdi(
        t = t_train,
        y = y_train,
        n_idx = n_idx,
        m_idata = m_idata,
        model_type = model_type,
        prior_level = prior_level,
        kind = "full"
    )
    
    # hdi for parameters
    fh.hdi_param(
        m_idata = m_idata,
        model_type = model_type,
        prior_level = prior_level
    )
    
    # save idata (for model comparison and predictions)
    m_idata.to_netcdf(f"../models_python/idata_{model_type}_{prior_level}.nc")