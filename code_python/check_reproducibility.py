'''
Simulation & EDA
'''

### python: packages & reproducibility ###
import numpy as np # core library for vector/matrix handling
import pandas as pd # core library for data-frames
import matplotlib.pyplot as plt # core plotting library
import pymc3 as pm # core library for bayesian analysis
import arviz as az # core plotting library for pyMC3
import xarray as xr # used for preprocessing
import theano.tensor as tt # only used for the covariation model
import seaborn as sns # only used for EDA
RANDOM_SEED = 42

### python: simulate data ###
np.random.seed(42) # reproducibility
n_id = 15 # 15 people (idx)
n_time = 15 # 15 time-steps (t)
idx = np.repeat(range(n_id), n_time)
a_real = np.random.normal(loc = 1, scale = 0.5, size = n_id) # intercept
b_real = np.random.normal(loc = 0.3, scale = 0.2, size = n_id) # beta
eps_real = np.random.normal(loc = 0, scale = 0.5, size = len(idx)) # error
t = np.resize(np.arange(n_time), n_time*n_id)
y = a_real[idx] + b_real[idx] * t + eps_real # outcome

# data frame 
d = pd.DataFrame({
    'idx': idx, 
    't': t,
    'y': y})

# train/test split
sort_val = np.sort(np.unique(d["t"].values))
min_val = min(sort_val)
length = int(round(len(sort_val)*0.6, 0)) # 60% in train. 
train = d[d["t"] <= min_val+length] # train data
test = d[d["t"] > min_val+length] # test data

### python: quick EDA ###
lm = sns.lmplot(x = "t", y = "y", hue = "idx", data = train) # seaborn
lm.fig.suptitle("Python: Quick EDA") # add title
plt.plot(); # show plot

''' model 1 '''

### python: preprocessing ###
# if you do not already have the train data from simulation
train = pd.read_csv("../data/train.csv")

# t & idx unique
t_unique = np.unique(train.t.values) # unique t values (with numpy)
idx_unique = np.unique(train.idx.values) # unique idx values (with numpy)

# get n of unique for shapes
n_time = len(t_unique) # length of t unique
n_idx = len(idx_unique) # length of idx unique

# create coords and dims 
coords = {
    'idx': idx_unique, 
    't': t_unique}

# take out dims 
dims = coords.keys() 

# data in correct format. 
t_train = train.t.values.reshape((n_idx, n_time)) # reshape format (numpy)
y_train = train.y.values.reshape((n_idx, n_time)) # reshape format (numpy)
idx_train = train.idx.values.reshape((n_idx, n_time)) # not relevant for pooled

# gather dataset with xarray
dataset = xr.Dataset(
    {'t_data': (dims, t_train),
    'y_data': (dims, y_train)},
    coords = coords)

### python: specify model and compile ###
def pooled(): 
    with pm.Model(coords=coords) as m_pooled:

        # shared variables
        t_ = pm.Data('t_shared', t_train, dims = dims)

        # specify priors for parameters & model error
        beta = pm.Normal("beta", mu = 0, sigma = 0.5)
        alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
        sigma = pm.HalfNormal("sigma", sigma = 0.5)

        # calculate mu
        mu = alpha + beta * t_

        # likelihood
        y_pred = pm.Normal(
            "y_pred",
            mu = mu, 
            sigma = sigma, 
            observed = y_train)

        # return the model                
        return m_pooled

# now run the function to compile the model
m_pooled = pooled()

### python: plate notation ###
pm.model_to_graphviz(m_pooled)

### python: prior predictive checks ###
# sample prior predictive 
with m_pooled:
    prior = pm.sample_prior_predictive(700, random_seed = RANDOM_SEED) 
    idata_pooled = az.from_pymc3(prior=prior)

# set up plot 
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(idata_pooled, # the idata
            group = "prior", # plot the prior
            num_pp_samples = 100, # how many draws
            ax = ax) # add to matplotlib ax. 

fig.suptitle("Python/pyMC3: prior predictive check")
fig.tight_layout()
plt.plot();

### python: sample posterior ###
with m_pooled: 
    idata_posterior = pm.sample(
        draws = 2000, 
        tune = 2000, 
        chains = 2,
        return_inferencedata = True,
        target_accept = .99,
        max_treedepth = 20,
        random_seed = RANDOM_SEED)
idata_pooled.extend(idata_posterior) # add to idata

### python: plot trace ###
az.plot_trace(idata_pooled)

### python: plot summary ###
az.summary(idata_pooled)

### python: sample posterior predictive ###
with m_pooled:
    post_pred = pm.sample_posterior_predictive(
        idata_pooled,
        var_names = [
            "y_pred",
            "alpha",
            "beta"],
        random_seed = RANDOM_SEED)
    idata_postpred = az.from_pymc3(
        posterior_predictive=post_pred)
idata_pooled.extend(idata_postpred) # add to idata

### plot posterior predictive ###
# set up matplotlib plot
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(idata_pooled, 
            num_pp_samples = 100,
            ax = ax)

fig.suptitle("Python/pyMC3: posterior predictive check")
fig.tight_layout()
plt.plot();

### python: plot hdi (fixed effects) ###
# take out posterior predictive from idata
post_pred = idata_pooled.posterior_predictive

# take out alpha and beta values
alpha = post_pred.alpha.values #shape: (1, 4.000)
beta = post_pred.beta.values #shape: (1, 4.000)

# calculate outcome based on alpha and beta 
outcome = (alpha + beta * t_unique[:, None]).T

# calculate mean y 
y_mean = outcome.mean(axis = 0)

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

### python: plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = idata_pooled.posterior_predictive

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

### python: HDI for parameters ###
# set up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))

# if you just want the figure this is enough
az.plot_forest(
        idata_pooled,
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

''' model 2 '''

def intercept(): 
    with pm.Model(coords=coords) as m_intercept: 

        # Inputs
        idx_ = pm.Data('idx_shared', idx_train, dims = dims)
        t_ = pm.Data('t_shared', t_train, dims = dims)

        # hyper-priors (group-level effects)
        alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma = 0.5)

        # fixed slope for beta
        beta = pm.Normal("beta", mu = 0, sigma = 0.5)

        # varying intercepts & slopes for alpha
        alpha_var = pm.Normal(
            "alpha_var", 
            mu = alpha, 
            sigma = sigma_alpha, 
            dims = "idx")

        # expected value per participant at each time-step
        mu = alpha_var[idx_] + beta * t_ 

        # model error
        sigma = pm.HalfNormal("sigma", sigma = 0.5)

        # likelihood
        y_pred = pm.Normal(
            "y_pred", 
            mu = mu, 
            sigma = sigma, 
            observed = y_train, 
            dims = dims)

        # return model
        return m_intercept

# now run the function to compile the model
m_intercept = intercept()

### python: plate notation ###
pm.model_to_graphviz(m_intercept)

### python: prior predictive checks ###
# sample prior predictive 
with m_intercept:
    prior = pm.sample_prior_predictive(700, random_seed = RANDOM_SEED) 
    idata_intercept = az.from_pymc3(prior=prior)

# set up plot 
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(idata_intercept, # the idata
            group = "prior", # plot the prior
            num_pp_samples = 100, # how many draws
            ax = ax) # add to matplotlib ax. 

fig.suptitle("Python/pyMC3: prior predictive check")
fig.tight_layout()
plt.plot();

### python: sample posterior ###
with m_intercept: 
    idata_posterior = pm.sample(
        draws = 2000, 
        tune = 2000, 
        chains = 2,
        return_inferencedata = True,
        target_accept = .99,
        max_treedepth = 20,
        random_seed = RANDOM_SEED)
idata_intercept.extend(idata_posterior) # add to idata

### python: plot trace ###
az.plot_trace(idata_intercept)

### python: plot summary ###
az.summary(idata_intercept)

### python: sample posterior predictive ###
with m_intercept:
    post_pred = pm.sample_posterior_predictive(
        idata_intercept,
        var_names = [
            "y_pred",
            "alpha",
            "beta"],
        random_seed = RANDOM_SEED)
    idata_postpred = az.from_pymc3(
        posterior_predictive=post_pred)
idata_intercept.extend(idata_postpred) # add to idata

### plot posterior predictive ###
# set up matplotlib plot
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(idata_intercept, 
            num_pp_samples = 100,
            ax = ax)

fig.suptitle("Python/pyMC3: posterior predictive check")
fig.tight_layout()
plt.plot();

### python: plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = idata_intercept.posterior_predictive

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

### python: plot hdi (fixed effects) ###
# take out posterior predictive from idata
post_pred = idata_intercept.posterior_predictive

# take out alpha and beta values
alpha = post_pred.alpha.values #shape: (1, 4.000)
beta = post_pred.beta.values #shape: (1, 4.000)

# calculate outcome based on alpha and beta 
outcome = (alpha + beta * t_unique[:, None]).T

# calculate mean y 
y_mean = outcome.mean(axis = 0)

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

### python: HDI for parameters ###
# set up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))

# if you just want the figure this is enough
az.plot_forest(
        idata_intercept,
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

''' model 3 '''

### python: additional preprocessing for covariation model ###
coords["param"] = ["alpha", "beta"]
coords["param_bis"] = ["alpha", "beta"]

def covariation(): 
    with pm.Model(coords=coords) as m_covariation: 

        # Inputs
        idx_ = pm.Data('idx_shared', idx_train, dims = ('idx', 't'))
        t_ = pm.Data('t_shared', t_train, dims = ('idx', 't'))


        # prior stddev in intercepts & slopes (variation across counties):
        sd_dist = pm.HalfNormal.dist(0.5) # distribution. 

        # get back standard deviations and rho:
        ## eta = 1: uniform (higher --> more weight on low cor.)
        ## n = 2: number of predictors
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", 
            n=2, 
            eta=2, 
            sd_dist=sd_dist, 
            compute_corr = True) 

        # priors for mean effects
        alpha = pm.Normal("alpha", mu = 1.5, sigma = 0.5)
        beta = pm.Normal("beta", mu = 0, sigma = 0.5)

        # population of varying effects
        alpha_beta = pm.MvNormal(
            "alpha_beta", 
            mu = tt.stack([alpha, beta]), 
            chol = chol, 
            dims=("idx", "param"))

        # expected value per participant at each time-step
        mu = alpha_beta[idx_, 0] + alpha_beta[idx_, 1] * t_

        # model error
        sigma = pm.HalfNormal("sigma", sigma = 0.5)

        # likelihood
        y_pred = pm.Normal(
            "y_pred", 
            mu = mu, 
            sigma = sigma, 
            observed = y_train, 
            dims = ('idx', 't'))

        # return the model
        return m_covariation

# now run the function to compile the model
m_covariation = covariation()

### python: plate notation ###
pm.model_to_graphviz(m_covariation)

### python: prior predictive checks ###
# sample prior predictive 
with m_covariation:
    prior = pm.sample_prior_predictive(700, random_seed = RANDOM_SEED) 
    idata_covariation = az.from_pymc3(prior=prior)

# set up plot 
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(idata_covariation, # the idata
            group = "prior", # plot the prior
            num_pp_samples = 100, # how many draws
            ax = ax) # add to matplotlib ax. 

fig.suptitle("Python/pyMC3: prior predictive check")
fig.tight_layout()
plt.plot();

### python: sample posterior ###
with m_covariation: 
    idata_posterior = pm.sample(
        draws = 2000, 
        tune = 2000, 
        chains = 2,
        return_inferencedata = True,
        target_accept = .99,
        max_treedepth = 20,
        random_seed = RANDOM_SEED)
idata_covariation.extend(idata_posterior) # add to idata

### python: plot trace ###
az.plot_trace(idata_covariation)

### python: plot summary ###
az.summary(idata_covariation)

### python: sample posterior predictive ###
with m_covariation:
    post_pred = pm.sample_posterior_predictive(
        idata_covariation,
        var_names = [
            "y_pred",
            "alpha",
            "beta"],
        random_seed = RANDOM_SEED)
    idata_postpred = az.from_pymc3(
        posterior_predictive=post_pred)
idata_covariation.extend(idata_postpred) # add to idata

### plot posterior predictive ###
# set up matplotlib plot
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc(idata_covariation, 
            num_pp_samples = 100,
            ax = ax)

fig.suptitle("Python/pyMC3: posterior predictive check")
fig.tight_layout()
plt.plot();

### python: plot hdi (fixed effects) ###
# take out posterior predictive from idata
post_pred = idata_covariation.posterior_predictive

# take out alpha and beta values
alpha = post_pred.alpha.values #shape: (1, 4.000)
beta = post_pred.beta.values #shape: (1, 4.000)

# calculate outcome based on alpha and beta 
outcome = (alpha + beta * t_unique[:, None]).T

# calculate mean y 
y_mean = outcome.mean(axis = 0)

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

### python: plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = idata_covariation.posterior_predictive

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

### python: HDI for parameters ###
# set up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))

# if you just want the figure this is enough
az.plot_forest(
        idata_covariation,
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

''' model comparison '''

### python: plot trace ###
az.plot_trace(idata_intercept)

### python: plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = idata_intercept.posterior_predictive

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

### python: model comparison ###
loo_overview = az.compare({
    "m_pooled": idata_pooled,
    "m_intercept": idata_intercept,
    "m_covariation": idata_covariation})

loo_overview

### python: preprocessing & sampling ###
# if you do not already have the test data from simulation
test = pd.read_csv("../data/test.csv")

# get unique values for shared. 
t_unique_test = np.unique(test.t.values)
idx_unique_test = np.unique(test.idx.values)

# get n unique for shapes. 
n_time_test = len(t_unique_test)
n_idx_test = len(idx_unique_test)

# new coords as well
prediction_coords = {
    'idx': idx_unique_test,
    't': t_unique_test
}

# test data in correct format. 
t_test = test.t.values.reshape((n_idx_test, n_time_test))
y_test = test.y.values.reshape((n_idx_test, n_time_test))
idx_test = test.idx.values.reshape((n_idx_test, n_time_test))

with m_covariation:
    pm.set_data({"t_shared": t_test, "idx_shared": idx_test})
    stl_pred = pm.fast_sample_posterior_predictive(
        idata_covariation.posterior, random_seed=RANDOM_SEED
    )
    az.from_pymc3_predictions(
        stl_pred, idata_orig=idata_covariation, inplace=True, coords=prediction_coords
    )
    
### python: plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = idata_covariation.posterior_predictive

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