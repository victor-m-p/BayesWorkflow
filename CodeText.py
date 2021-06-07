### import packages ###
import streamlit as st

### python functions ###

# python reproducibility
def py_reproducibility():
    py_code = f'''
## python: packages & reproducibility ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pymc3 as pm
import arviz as az
import xarray as xr
import seaborn as sns
RANDOM_SEED = 42
'''
    return py_code

# python preprocessing 
def py_preprocessing(): 
    py_code = f'''
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
coords = {{
    'idx': idx_unique,
    't': t_unique}}

# take out dims 
dims = coords.keys()

# data in correct format. 
t_train = train.t.values.reshape((n_idx, n_time))
y_train = train.y.values.reshape((n_idx, n_time))
idx_train = train.idx.values.reshape((n_idx, n_time)) # not relevant for pooled

# gather dataset 
dataset = xr.Dataset(
    {{'t_data': (dims, t_train),
    'y_data': (dims, y_train)}},
    coords = coords)
'''
    return py_code

# pooled model
def py_pooled(model_name, sigma_choice):
    py_code = f'''
### python: specify model and compile ###
with pm.Model(coords=coords) as {model_name}:
    
    # shared variables
    t_ = pm.Data('t_shared', 
                t_train, 
                dims = dims)
    
    # specify priors for parameters & model error
    beta = pm.Normal("beta", 
                    mu = 0, 
                    sigma = {sigma_choice})
    alpha = pm.Normal("alpha", 
                    mu = 1.5, 
                    sigma = {sigma_choice})
    sigma = pm.HalfNormal("sigma", 
                        sigma = {sigma_choice})
    
    # calculate mu
    mu = alpha + beta * t_
    
    # likelihood
    y_pred = pm.Normal("y_pred", 
                    mu = mu, 
                    sigma = sigma, 
                    observed = y_train)
'''
    return py_code

def py_multilevel(model_name, sigma_choice): 
    py_code = f'''
with pm.Model(coords=coords) as {model_name}: 
    
    # Inputs
    idx_ = pm.Data('idx_shared', idx_train, dims = dims)
    t_ = pm.Data('t_shared', t_train, dims = dims)

    # hyper-priors
    alpha = pm.Normal("alpha", mu = 1.5, sigma = {sigma_choice})
    alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = {sigma_choice})
    beta = pm.Normal("beta", mu = 0, sigma = {sigma_choice})
    beta_sigma = pm.HalfNormal("beta_sigma", sigma = {sigma_choice})
    
    # varying intercepts & slopes
    alpha_varying = pm.Normal("alpha_varying", mu = alpha, sigma = alpha_sigma, dims = "idx")
    beta_varying = pm.Normal("beta_varying", mu = beta, sigma = beta_sigma, dims = "idx")
    
    # expected value per participant at each time-step
    mu = alpha_varying[idx_] + beta_varying[idx_] * t_

    # model error
    sigma = pm.HalfNormal("sigma", sigma = {sigma_choice})
    
    # likelihood
    y_pred = pm.Normal("y_pred", mu = mu, sigma = sigma, observed = y_train, dims = dims)
    '''
    return py_code

def py_student(model_name, sigma_choice):
    py_code = f'''
with pm.Model(coords=coords) as {model_name}: 
    
    # Inputs
    idx_ = pm.Data('idx_shared', idx_train, dims = dims)
    t_ = pm.Data('t_shared', t_train, dims = dims)

    # hyper-priors
    alpha = pm.Normal("alpha", mu = 1.5, sigma = {sigma_choice})
    alpha_sigma = pm.HalfNormal("alpha_sigma", sigma = {sigma_choice})
    beta = pm.Normal("beta", mu = 0, sigma = {sigma_choice})
    beta_sigma = pm.HalfNormal("beta_sigma", sigma = {sigma_choice})
    
    # varying intercepts & slopes
    alpha_varying = pm.Normal("alpha_varying", mu = alpha, sigma = alpha_sigma, dims = "idx")
    beta_varying = pm.Normal("beta_varying", mu = beta, sigma = beta_sigma, dims = "idx")
    
    # expected value per participant at each time-step
    mu = alpha_varying[idx_] + beta_varying[idx_] * t_

    # nu
    v = pm.Gamma("v", alpha = 2, beta = 0.1)
    
    # model error
    sigma = pm.HalfNormal("sigma", sigma = {sigma_choice})
    
    # likelihood
    y_pred = pm.StudentT("y_pred", nu = v, mu = mu, sigma = sigma, observed = y_train, dims = dims)
    '''
    return py_code
    
# plate
def py_plate(model_name): # m_pooled, m_multilevel, m_student
    py_code = f'''
### python: plate notation ###
pm.model_to_graphviz({model_name})
'''
    return py_code

# prior predictive
def py_pp(model_name, idata_name): # m_pooled, m_multilevel, m_student
    py_code = f'''
### python: prior predictive checks ###
# sample prior predictive 
with {model_name}:
    prior = pm.sample_prior_predictive(700, random_seed = RANDOM_SEED) 
    {idata_name} = az.from_pymc3(prior=prior)
    
# set up plot 
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc({idata_name}, # the idata
            group = "prior", # plot the prior
            num_pp_samples = 100, # how many draws
            ax = ax) # add to matplotlib ax. 

fig.suptitle("Python/pyMC3: prior predictive check")
fig.tight_layout()
plt.plot();
'''
    return py_code

# sample posterior
def py_sample(model_name, idata_name):
    py_code = f'''
### python: sample posterior ###
with {model_name}: 
    idata_posterior = pm.sample(
        draws = 2000, 
        tune = 2000, 
        chains = 2,
        return_inferencedata = True,
        target_accept = .99,
        max_treedepth = 20,
        random_seed = RANDOM_SEED)
{idata_name}.extend(idata_posterior) # add to idata
'''
    return py_code
    
# trace
def py_trace(idata_name):
    py_code = f'''
### python: plot trace ###
az.plot_trace({idata_name})
'''
    return py_code

# summary
def py_summary(idata_name): 
    py_code = f'''
### python: plot summary ###
az.summary({idata_name})
'''
    return py_code

# posterior predictive
def py_post_pred(model_name, idata_name):
    py_code = f'''
### python: sample posterior predictive ###
with {model_name}:
    post_pred = pm.sample_posterior_predictive(
        {idata_name},
        var_names = [
            "y_pred",
            "alpha",
            "beta"],
        random_seed = RANDOM_SEED)
    idata_postpred = az.from_pymc3(
        posterior_predictive=post_pred)
{idata_name}.extend(idata_postpred) # add to idata

### plot posterior predictive ###
# set up matplotlib plot
fig, ax = plt.subplots()

# if you just want the figure then this is enough
az.plot_ppc({idata_name}, 
            num_pp_samples = 100,
            ax = ax)

fig.suptitle("Python/pyMC3: posterior predictive check")
fig.tight_layout()
plt.plot();
'''
    return py_code

# hdi
def py_hdi_data_fixed(hdi_type, idata_name): 
    py_code = f'''
### plot hdi (fixed effects) ###
# take posterior predictive out of idata for convenience
ppc = {idata_name}.posterior_predictive

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
    fill_kwargs= {{'alpha': 0.4, "label": "80% HPD intervals"}},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {{'alpha': 0.3, "label": "95% HDI intervals"}},
    hdi_prob = high)

# add legend, title and tight layout. 
ax.legend()
fig.suptitle("Python/pyMC3: Prediction Intervals ({hdi_type})")
fig.tight_layout()

# plot it 
plt.plot();
'''
    return py_code 

def py_hdi_data_full(hdi_type, idata_name): 
    py_code = f'''
### plot hdi (full uncertainty) ###
# take posterior predictive out of idata for convenience
ppc = {idata_name}.posterior_predictive

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
    fill_kwargs= {{'alpha': 0.4, "label": "80% HDI intervals"}},
    hdi_prob = low)

# plot higher interval
az.plot_hdi(
    t_unique,
    outcome,
    ax = ax,
    fill_kwargs = {{'alpha': 0.3, "label": "95% HDI intervals"}},
    hdi_prob = high)

# add legend, title and tight layout. 
ax.legend()
fig.suptitle("Python/pyMC3: Prediction Intervals ({hdi_type})")
fig.tight_layout()

# plot it 
plt.plot();
'''
    return py_code

def py_hdi_param(idata_name): 
    py_code = f'''
### HDI for parameters ###
# set up matplotlib plot
fig, ax = plt.subplots(figsize = (10, 7))

# if you just want the figure this is enough
az.plot_forest(
        {idata_name},
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
'''
    return py_code

def py_pred_prep(model_name, idata_name): 
    py_code = f'''
### python: preprocessing & sampling ###
# load test data
test = pd.read_csv("../data/test.csv")

# get unique values for shared. 
t_unique_test = np.unique(test.t.values)
idx_unique_test = np.unique(test.idx.values)

# get n unique for shapes. 
n_time_test = len(t_unique_test)
n_idx_test = len(idx_unique_test)

# new coords as well
prediction_coords = {{
    'idx': idx_unique_test,
    't': t_unique_test
}}

# test data in correct format. 
t_test = test.t.values.reshape((n_idx_test, n_time_test))
y_test = test.y.values.reshape((n_idx_test, n_time_test))
idx_test = test.idx.values.reshape((n_idx_test, n_time_test))

with {model_name}:
    pm.set_data({{"t_shared": t_test, "idx_shared": idx_test}})
    stl_pred = pm.fast_sample_posterior_predictive(
        {idata_name}.posterior, random_seed=RANDOM_SEED
    )
    az.from_pymc3_predictions(
        stl_pred, idata_orig={idata_name}, inplace=True, coords=prediction_coords
    )
    '''
    return py_code

def py_sim(): 
    py_code = f'''
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
d = pd.DataFrame({{
    'idx': idx, 
    't': t,
    'y': y}})
    
# train/test split
sort_val = np.sort(np.unique(d["t"].values))
min_val = min(sort_val)
length = int(round(len(sort_val)*0.6, 0)) # 60% in train. 
train = d[d["t"] <= min_val+length] # train data
test = d[d["t"] > min_val+length] # test data

# save data
train.to_csv("../data/train.csv", index = False)
test.to_csv("../data/test.csv", index = False)
    '''
    return py_code

def py_EDA(): 
    py_code = f'''
### python: quick EDA ###
lm = sns.lmplot(x = "t", y = "y", hue = "idx", data = train)
lm.fig.suptitle("Python: Quick EDA") # add title
    '''
    return py_code 

### R functions ###
# r reproducibility
def R_reproducibility():
    R_code = f'''
### R: packages & reproducibility ###
pacman::p_load(
    tidyverse, 
    brms,
    modelr,
    tidybayes,
    bayesplot)
RANDOM_SEED = 42
'''
    return R_code

# r preprocessing
def R_preprocessing():
    R_code = f'''
### R: preprocessing ###
train <- read_csv("../data/train.csv") %>%
    mutate(idx = as_factor(idx))
'''
    return R_code

# pooled model
def R_pooled(model_name, model_formula, prior_name, sigma_choice): 
    R_code = f'''
### R: specify model & compile ###
# formula 
{model_formula} <- bf(y ~ 1 + t) # complete pooling 

# set priors --> can use get_prior() if in doubt. 
{prior_name} <- c(
    prior(normal(0, {sigma_choice}), class = b),
    prior(normal(1.5, {sigma_choice}), class = Intercept),
    prior(normal(0, {sigma_choice}), class = sigma))

# compile model & sample prior
{model_name} <- brm(
    formula = {model_formula},
    family = gaussian,
    data = train,
    prior = {prior_name},
    sample_prior = "only",
    backend = "cmdstanr",
    seed = RANDOM_SEED)
'''
    return R_code

def R_multilevel(model_name, model_formula, prior_name, sigma_choice):
    R_code = f'''
### R: specify model & compile ###
# formula 
{model_formula} <- bf(y ~ 1 + t + (1+t|idx)) # random eff. structure 
    
# set priors --> can use get_prior() if in doubt. 
{prior_name} <- c(
    prior(normal(0, {sigma_choice}), class = b),
    prior(normal(1.5, {sigma_choice}), class = Intercept),
    prior(normal(0, {sigma_choice}), class = sd), # new
    prior(normal(0, {sigma_choice}), class = sigma),
    prior(lkj(1), class = cor) # new
)

# compile model & sample prior
{model_name} <- brm(
    formula = {model_formula},
    family = gaussian,
    data = train,
    prior = {prior_name},
    sample_prior = "only",
    backend = "cmdstanr")
    '''
    return R_code

def R_student(model_name, model_formula, prior_name, sigma_choice):
    R_code = f'''
### R: specify model & compile ###
# formula 
{model_formula} <- bf(y ~ 1 + t + (1+t|idx)) # random eff. structure 

# set priors --> can use get_prior() if in doubt. 
prior_student_specific <- c(
    prior(normal(0, {sigma_choice}), class = b),
    prior(normal(1.5, {sigma_choice}), class = Intercept),
    prior(normal(0, {sigma_choice}), class = sd),
    prior(normal(0, {sigma_choice}), class = sigma),
    prior(lkj(1), class = cor),
    prior(gamma(2, 0.1), class = nu) # new. 
)

# compile model & sample prior
{model_name} <- brm(
    formula = {model_formula},
    family = student, # student-t likelihood function
    data = train,
    prior = {prior_name},
    sample_prior = "only",
    backend = "cmdstanr",
    seed = RANDOM_SEED)
    '''
    return R_code
    
# prior predictive
def R_pp(model_name):
    R_code = f'''
### R: Prior predictive checks ###
pp_check({model_name}, 
        nsamples = 100) +
    labs(title = "R/brms: prior predictive check") 
'''
    return R_code

# sample posterior
def R_sample(model_name, model_formula, model_family, prior_name):
    R_code = f'''
### R: sample posterior ###
{model_name} <- brm(
    formula = {model_formula},
    family = {model_family},
    data = train,
    prior = {prior_name},
    sample_prior = TRUE, # only difference. 
    backend = "cmdstanr",
    chains = 2,
    cores = 4,
    iter = 4000, 
    warmup = 2000,
    threads = threading(2), # not sure this can be done in pyMC3
    control = list(adapt_delta = .99,
                    max_treedepth = 20),
    seed = RANDOM_SEED)
'''
    return R_code

# trace 
def R_trace(model_name):
    R_code = f'''
### R: plot trace ###
plot({model_name},
    N = 10) # N param per plot. 
'''
    return R_code

# summary
def R_summary(model_name):
    R_code = f'''
### R: get summary (not displayed) ###
summary({model_name})
'''
    return R_code

# posterior predictive
def R_post_pred(model_name):
    R_code = f'''
### R: Posterior predictive checks ###
pp_check({model_name}, 
        nsamples = 100) + 
labs(title = "R/brms: posterior predictive check") 
'''
    return R_code

# hdi for fixed effects
### R: Plot HDI (fixed effects) ###

# type = .prediction, .fixed 
# data = train, test. 
def R_hdi_data_pool(model_name, pred_type, data_type, function, title): 
    R_code = f'''
### R: HDI prediction intervals ###
{data_type} %>%
    data_grid(t = seq_range(t, n = 100)) %>%
    {function}({model_name}) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = {pred_type}), 
                    .width = c(.95, .8), # HDI intervals
                    color = "#08519C",
                    point_interval = median_hdi) + 
    geom_jitter(data = {data_type}, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() + 
    ggtitle("R/brms: Prediction intervals ({title})")
'''
    return R_code

def R_hdi_fixed_groups(model_name, pred_type, data_type, function, title): 
    R_code = f'''
### R: HDI prediction intervals ###
{data_type} %>%
    data_grid(t = seq_range(t, n = 100), idx) %>%
    {function}({model_name}) %>%
    ggplot(aes(x = t, y = y, re_formula = NA)) + 
    stat_lineribbon(aes(y = {pred_type}), 
                    .width = c(.95, .8), # HDI intervals
                    color = "#08519C",
                    point_interval = median_hdi) + 
    geom_jitter(data = {data_type}, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() + 
    ggtitle("R/brms: Prediction intervals ({title})")
'''
    return R_code

def R_hdi_full_groups(model_name, pred_type, data_type, function, title): 
    R_code = f'''
### R: HDI prediction intervals ###
{data_type} %>%
    data_grid(t = seq_range(t, n = 100), idx) %>%
    {function}({model_name}) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = {pred_type}), 
                    .width = c(.95, .8), # HDI intervals
                    color = "#08519C",
                    point_interval = median_hdi) + 
    geom_jitter(data = {data_type}, 
                color = "navyblue", 
                shape = 1,
                alpha = 0.5, 
                size = 2, 
                width = 0.1) + 
    scale_fill_brewer() + 
    ggtitle("R/brms: Prediction intervals ({title})")
'''
    return R_code

# hdi vs. data
def R_hdi_param(model_name): 
    R_code = f'''
### R: HDI for parameters ###
mcmc_areas(
{model_name},
pars = c("b_Intercept",
        "b_t",
        "sigma"),
prob = 0.8, # 80% intervals
prob_outer = 0.99, # 99%
point_est = "mean") + # or median?
ggtitle("R/brms: HDI intervals for parameters")
'''
    return R_code

def R_pred_prep(): 
    R_code = f'''
### R: read test data ###
test <- read_csv("../data/test.csv") %>%
    mutate(idx = as_factor(idx))
    '''
    return R_code

def R_EDA(): 
    R_code = f'''
ggplot(data = train, aes(x = t, y = y, color = idx)) +
    geom_point() + 
    geom_smooth(method = "lm", aes(fill = idx), alpha = 0.2) + 
    ggtitle("R: Quick EDA")
    '''
    return R_code