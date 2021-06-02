
## whole thing for "multilevel". 

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
y_train = train.y.values
idx_train, idx_unique = pd.factorize(train["idx"], sort=True)
n = len(train.idx) # obs id: unique.
t_unique = np.unique(t_train)

coords2 = {
    "obs_id": np.arange(n), # unique id.
    "idx": idx_unique, # unique idx. 
    "t_unique": t_unique
}

with pm.Model(coords = coords2) as m2: 
    
    # shared variables
    t_shared = pm.Data('t_shared', t_train, dims = "obs_id")
    idx_shared = pm.Data('idx_shared', idx_train, dims = "obs_id")
    
    # hyper-priors
    alpha_hyper = pm.Normal("alpha_hyper", mu = 1.5, sigma = 0.5)
    alpha_sigma_hyper = pm.HalfNormal("alpha_sigma_hyper", sigma = 0.5)
    beta_hyper = pm.Normal("beta_hyper", mu = 0, sigma = 0.5)
    beta_sigma_hyper = pm.HalfNormal("beta_sigma_hyper", sigma = 0.5)
    
    # varying intercepts & slopes
    alpha = pm.Normal("alpha", mu = alpha_hyper, sigma = alpha_sigma_hyper, dims = "idx")
    beta = pm.Normal("beta", mu = beta_hyper, sigma = beta_sigma_hyper, dims = "idx")
    
    # expected value per participant at each time-step
    mu = alpha[idx_shared] + beta[idx_shared] * t_shared
    
    # model error
    sigma = pm.HalfNormal("sigma", sigma = 0.5)
    
    # store fitted 
    # fitted = pm.Normal("fitted", mu = alpha + beta * t_shared, sigma)
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = "obs_id")


with m2: 
    m_idata2 = pm.sample(
        2000, 
        tune=2000, 
        random_seed=32, 
        return_inferencedata=True
    )
    
# try posterior predictive
with m2:
    post_pred2 = pm.sample_posterior_predictive(
        m_idata2,
        var_names = ["y_pred", "alpha", "beta"])
    idata_aux2 = az.from_pymc3(posterior_predictive=post_pred2)
    
m_idata2.extend(idata_aux2)

# now we mean over the different alpha and beta
# check that this is the same? as taking hyper.

# take out posterior predictive for ease. 
t_shared = m_idata2.constant_data.t_shared
ppc = m_idata2.posterior_predictive.assign_coords(t_shared=t_shared)

# take out alpha and beta, mean over idx. 
alpha = ppc["alpha"].mean(dim = "idx").values
beta = ppc["beta"].mean(dim = "idx").values

# calculate mu. 
mu_pp = (alpha + beta * t_unique[:, None]).T

# set up plot 
fig, ax = plt.subplots()

# plot data. 
ax.plot(t_train, y_train, 'o')

# plot 94% HDI. 
az.plot_hdi(
    t_unique,
    mu_pp
)

## just doing it manually for y_pred now. 
test = ppc["y_pred"].values
test.shape
test.shape
test2 = test.flatten()
new_shape = test2.reshape((60000, 10))

az.plot_hdi(
    t_unique,
    new_shape
)

# the full plot then
# plot data. 
fig, ax = plt.subplots()
ax.plot(t_train, y_train, 'o')

# plot 94% HDI. 
az.plot_hdi(
    t_unique,
    mu_pp,
    ax = ax,
    fill_kwargs={"alpha": 0.4, "label": "Mean outcome 94% HPD"}
)

az.plot_hdi(
    t_unique,
    new_shape,
    ax = ax,
    fill_kwargs={'alpha': 0.3, "label": "Prediction intervals 94% HPD"}
)

## HDI intervals
az.plot_forest(
    m_idata2,
    var_names=["alpha_hyper", "beta_hyper", "sigma"]
);


## predictions for a g
## load test data 
with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)
    
# take out the vectors
t_test = test.t.values
y_test = test.y.values
idx_test, idx_unique = pd.factorize(test["idx"], sort=True)
n = len(test.idx) # obs id: unique.
t_unique = np.unique(t_test)
t_unique_len = len(t_unique)

prediction_coords = {"obs_id": np.arange(n)}
with m2:
    pm.set_data({"t_shared": t_test, "idx_shared": idx_test})
    stl_pred = pm.fast_sample_posterior_predictive(
        m_idata2.posterior, random_seed=32
    )
    az.from_pymc3_predictions(
        stl_pred, idata_orig=m_idata2, inplace=True, coords=prediction_coords
    )


t_shared = m_idata2.predictions_constant_data.t_shared
pred = m_idata2.predictions.assign_coords(t_shared=t_shared)

## take out predictions in right format
y_pred = pred["y_pred"].values.reshape((-1, t_unique_len)) # -1 = undefined, numpy infers it. 
y_pred.shape
# mean predictions
y_mean = y_pred.mean(axis = 0)

## plot the real new data
fig, ax = plt.subplots()
ax.plot(t_test, y_test, 'o')
ax.plot(t_unique, y_mean)

# prediction interval 94 HDI. 
az.plot_hdi(
    t_unique,
    y_pred,
    ax = ax,
    fill_kwargs={'alpha': 0.3, "label": "Prediction intervals 94% HPD"}
)

## HDI intervals
az.plot_forest(
    m_idata2,
    var_names=["alpha_hyper", "beta_hyper", "sigma"]
);

