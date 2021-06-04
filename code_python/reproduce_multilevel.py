'''
reproduced based on code copied from streamlit to test.
'''

## python: packages & reproducibility ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pymc3 as pm
import arviz as az
import xarray as xr
RANDOM_SEED = 42

### python: preprocessing ###
# read data
train = pd.read_csv("../data/train.csv")

# t & idx unique
t_unique = np.unique(train.t.values)
idx_unique = np.unique(train.idx.values)

# get n of unique for shapes
n_time = len(t_unique)
n_idx = len(idx_unique)

# create coords and dims 
coords = {
    'idx': idx_unique,
    't': t_unique}

# take out dims 
dims = coords.keys()

# data in correct format. 
t_train = train.t.values.reshape((n_idx, n_time))
y_train = train.y.values.reshape((n_idx, n_time))
idx_train = train.idx.values.reshape((n_idx, n_time)) # not relevant for pooled

# gather dataset 
dataset = xr.Dataset(
    {'t_data': (dims, t_train),
    'y_data': (dims, y_train)},
    coords = coords)

with pm.Model(coords=coords) as m_multilevel: 

    # Inputs
    idx_ = pm.Data('idx_shared', idx_train, dims = dims)
    t_ = pm.Data('t_shared', t_train, dims = dims)

    # hyper-priors
    alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
    alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = 0.5)
    beta = pm.Normal("beta", mu = 0, sigma = 0.5)
    beta_sigma = pm.HalfNormal("beta_sigma", sigma = 0.5)

    # varying intercepts & slopes
    alpha_varying = pm.Normal("alpha_varying", mu = alpha, sigma = alpha_sigma, dims = "idx")
    beta_varying = pm.Normal("beta_varying", mu = beta, sigma = beta_sigma, dims = "idx")

    # expected value per participant at each time-step
    mu = alpha_varying[idx_] + beta_varying[idx_] * t_

    # model error
    sigma = pm.HalfNormal("sigma", sigma = 0.5)

    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = dims)
    
### python: plate notation ###
pm.model_to_graphviz(m_multilevel)

### python: prior predictive checks ###
# sample prior predictive 
with m_multilevel:
    prior = pm.sample_prior_predictive(700) 
    m_idata = az.from_pymc3(prior=prior)

# set up plot 
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(m_idata, # the idata
            group = "prior", # plot the prior
            num_pp_samples = 100, # how many draws
            ax = ax) # add to matplotlib ax. 

fig.suptitle("Python/pyMC3: prior predictive check")
fig.tight_layout()
plt.plot();

### python: sample posterior ###
with m_multilevel: 
    idata_posterior = pm.sample(
        draws = 2000, 
        tune = 2000, 
        chains = 2,
        return_inferencedata = True,
        target_accept = .99,
        max_treedepth = 20,
        random_seed = RANDOM_SEED)
m_idata.extend(idata_posterior) # add to idata

### python: plot trace ###
az.plot_trace(m_idata)

### python: plot summary ###
az.summary(m_idata)

### python: sample posterior predictive ###
with m_multilevel:
    post_pred = pm.sample_posterior_predictive(
        m_idata,
        var_names = [
            "y_pred",
            "alpha",
            "beta"])
    idata_postpred = az.from_pymc3(
        posterior_predictive=post_pred)
m_idata.extend(idata_postpred) # add to idata

### plot posterior predictive ###
# set up matplotlib plot
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(m_idata, 
            num_pp_samples = 100,
            ax = ax)

fig.suptitle("Python/pyMC3: posterior predictive check")
fig.tight_layout()
plt.plot();

### plot hdi (fixed effects) ###
# take posterior predictive out of idata for convenience
ppc = m_idata.posterior_predictive

# take out predictions (mean over chains). 
y_pred = ppc["y_pred"].mean(axis = 0).values

# calculate mean y predicted (mean over draws and idx)
y_mean = y_pred.mean(axis = (0, 1))

# calculate the outcome based on fixed effects
outcome = (ppc.alpha.values + ppc.beta.values * t_unique[:, None]).T

# set-up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange", #aesthetics
    alpha = 0.5 # aesthetics
    )

# plot mean
ax.plot(
    t_unique, 
    y_mean,
    color = "darkorange" # aesthetics
    )

# set HDI intervals 
high, low = (.95, .8) 

# plot lower interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs= {'alpha': 0.4, "label": "80% HPD intervals"},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": "95% HDI intervals"},
    hdi_prob = high)

# add legend, title and tight layout. 
ax.legend()
fig.suptitle("Python/pyMC3: Prediction Intervals (fixed)")
fig.tight_layout()

# plot it 
plt.plot();

### plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = m_idata.posterior_predictive

# take out predictions (mean over chains). 
y_pred = ppc["y_pred"].mean(axis = 0).values

# calculate mean y predicted (mean over draws and idx)
y_mean = y_pred.mean(axis = (0, 1))

# THE DIFFERENCE: base it on the actual predictions of the full model 
outcome = y_pred.reshape((4000*n_idx, n_time)) # 4000 = 2000 (draws) * 2 (chains)

# set-up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))  

# plot data
ax.scatter(
    t_train, 
    y_train,
    color = "darkorange", #aesthetics
    alpha = 0.5 # aesthetics
    )

# plot mean
ax.plot(
    t_unique, 
    y_mean,
    color = "darkorange" # aesthetics
    )

# set HDI intervals 
high, low = (.95, .8) 

# plot lower interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs= {'alpha': 0.4, "label": "80% HDI intervals"},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {'alpha': 0.3, "label": "95% HDI intervals"},
    hdi_prob = high)

# add legend, title and tight layout. 
ax.legend()
fig.suptitle("Python/pyMC3: Prediction Intervals (full)")
fig.tight_layout()

# plot it 
plt.plot();

### HDI for parameters ###
# set up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))

# if you just want the figure this is enough
az.plot_forest(
        m_idata,
        var_names=["alpha", "beta", "sigma"], # the parameters we care about
        combined=True, # combine chains 
        kind='ridgeplot', # instead of default which does not show distribution
        ridgeplot_truncate=False, # do show the tails 
        hdi_prob = .8, # hdi prob .8 here. 
        ridgeplot_alpha = 0.5, # looks prettier
        ridgeplot_quantiles = [0.5], # show mean
        ax = ax
        ) 

fig.suptitle("Python/pyMC3: HDI intervals for parameters")
fig.tight_layout()
plt.plot();