## packages ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pymc3 as pm
import seaborn as sns
import arviz as az
import fun_models as fm
import fun_helper as fh
import dataframe_image as dfi
import xarray as xr
import theano.tensor as tt

# https://docs.pymc.io/notebooks/multilevel_modeling.html

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
idx_train = train.idx.values.reshape((n_idx, n_time))

# gather dataset 
dataset = xr.Dataset(
    {'t_data': (dims, t_train),
    'y_data': (dims, y_train)},
    coords = coords)

# population of varying effects:
coords["param"] = ["alpha", "beta"]
coords["param_bis"] = ["alpha", "beta"]

#pars = np.array(["alpha", "beta"])
def student(t, idx, y, coords, dims, sigma = 0.5): 
    
    with pm.Model(coords=coords) as m: 
        
        # Inputs
        idx_ = pm.Data('idx_shared', idx, dims = ('idx', 't'))
        t_ = pm.Data('t_shared', t, dims = ('idx', 't'))

        
        # prior stddev in intercepts & slopes (variation across counties):
        sd_dist = pm.HalfNormal.dist(sigma) # distribution. 

        # get back standard deviations and rho:
        ## eta = 1: uniform (higher --> more weight on low cor.)
        ## n = 2: number of predictors
        chol, corr, stds = pm.LKJCholeskyCov("chol", n=2, eta=1, sd_dist=sd_dist, compute_corr = True) 

        # priors for mean effects
        alpha = pm.Normal("alpha", mu = 1.5, sigma = sigma)
        beta = pm.Normal("beta", mu = 0, sigma = sigma)
        
        # population of varying effects
        alpha_beta = pm.MvNormal("alpha_beta", 
                                 mu = tt.stack([alpha, beta]), 
                                 chol = chol, 
                                 dims=("idx", "param"))
    
        # expected value per participant at each time-step
        mu = alpha_beta[idx_, 0] + alpha_beta[idx_, 1] * t_

        # model error
        sigma = pm.HalfNormal("sigma", sigma = sigma)
        
        # likelihood
        y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y, dims = ('idx', 't'))
        
    return m

# compile 
m = student(t_train, idx_train, y_train, coords, dims, sigma = 0.5)


# load netcdf
m_idata = az.from_netcdf("../models_python/correlation3.nc")

m_idata.posterior

# graphviz
pm.model_to_graphviz(m) 

# pp checks
az.plot_ppc(m_idata, group = "prior", num_pp_samples = 100)

# posterior predictive 
az.plot_ppc(m_idata, num_pp_samples = 100)

# trace (some warning here)
az.plot_trace(m_idata)

# crazy summary
az.summary(m_idata)

# HDI full #
# unpack tuple & get unique t. 
high, low = (.95, .8)
t_unique = np.unique(t_train)
n_time = len(t_unique)

# take out ppc 
ppc = m_idata.posterior_predictive
y_pred = ppc["y_pred"].mean(axis = 0).values
y_mean = y_pred.mean(axis = (0, 1))
outcome = y_pred.reshape((4000*n_idx, n_time))

# set-up plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange",
    alpha = 0.5)

# plot mean
ax.plot(t_unique, y_mean,
        color = "darkorange")

# plot lower interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": f"{low*100}% HPD intervals"},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": f"{high*100}% HDI intervals"},
    hdi_prob = high)
    


# HDI (fixed) #
# take out ppc 
ppc = m_idata.posterior_predictive
alpha = ppc.alpha.values #shape: (1, 4.000)
beta = ppc.beta.values
outcome = (alpha + beta * t_unique[:, None]).T
y_mean = outcome.mean(axis = 0)

# set-up plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange",
    alpha = 0.5)

# plot mean
ax.plot(t_unique, y_mean,
        color = "darkorange")

# plot lower interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs={'alpha': 0.4, "label": f"{low*100}% HPD intervals"},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": f"{high*100}% HDI intervals"},
    hdi_prob = high)
    

# sample prior pred
with m:
    prior = pm.sample_prior_predictive(700, random_seed = RANDOM_SEED) # small, but only for pp. 
    m_idata = az.from_pymc3(prior=prior)

# sample posterior
with m:
    idata_posterior = pm.sample(
        draws = n_draws, 
        tune = n_tune, 
        chains = n_chains,
        return_inferencedata = True,
        #target_accept = target_accept,
        #max_treedepth = max_treedepth,
        random_seed = RANDOM_SEED)
        #idata_kwargs = {"dims": {"chol_stds": ["param"], "chol_corr": ["param", "param_bis"]}})
m_idata.extend(idata_posterior)

# sample posterior pred
with m:
    post_pred = pm.sample_posterior_predictive(
        m_idata,
        var_names = [
            "y_pred",
            "alpha",
            "beta"],
        random_seed = RANDOM_SEED)
    idata_postpred = az.from_pymc3(posterior_predictive=post_pred)
m_idata.extend(idata_postpred)

m_idata.to_netcdf("../models_python/correlation4.nc")
