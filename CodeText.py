### import packages ###
import streamlit as st

### python functions ###

# python reproducibility
def py_reproducibility():
    py_code = f'''
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
'''
    return py_code

# python preprocessing 
def py_preprocessing(): 
    py_code = f'''
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
coords = {{
    'idx': idx_unique, 
    't': t_unique}}

# take out dims 
dims = coords.keys() 

# data in correct format. 
t_train = train.t.values.reshape((n_idx, n_time)) # reshape format (numpy)
y_train = train.y.values.reshape((n_idx, n_time)) # reshape format (numpy)
idx_train = train.idx.values.reshape((n_idx, n_time)) # not relevant for pooled

# gather dataset with xarray
dataset = xr.Dataset(
    {{'t_data': (dims, t_train),
    'y_data': (dims, y_train)}},
    coords = coords)

'''
    return py_code

def py_preprocessing_cov(): 
    py_code = f'''
### python: additional preprocessing for covariation model ###
coords["param"] = ["alpha", "beta"]
coords["param_bis"] = ["alpha", "beta"]
'''
    return py_code

# pooled model
def py_pooled(model_name, sigma_choice):
    py_code = f'''
### python: specify model and compile ###
def pooled(): 
    with pm.Model(coords=coords) as {model_name}:
        
        # shared variables
        t_ = pm.Data('t_shared', t_train, dims = dims)
        
        # specify priors for parameters & model error
        beta = pm.Normal("beta", mu = 0, sigma = {sigma_choice})
        alpha = pm.Normal("alpha", mu = 1.5, sigma = {sigma_choice})
        sigma = pm.HalfNormal("sigma", sigma = {sigma_choice})
        
        # calculate mu
        mu = alpha + beta * t_
        
        # likelihood
        y_pred = pm.Normal(
            "y_pred",
            mu = mu, 
            sigma = sigma, 
            observed = y_train)
                        
        # return the model                
        return {model_name}

# now run the function to compile the model
{model_name} = pooled()
'''
    return py_code

def py_intercept(model_name, sigma_choice): 
    py_code = f'''
def intercept(): 
    with pm.Model(coords=coords) as {model_name}: 
            
        # Inputs
        idx_ = pm.Data('idx_shared', idx_train, dims = dims)
        t_ = pm.Data('t_shared', t_train, dims = dims)

        # hyper-priors (group-level effects)
        alpha = pm.Normal("alpha", mu = 1.5, sigma = {sigma_choice})
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma = {sigma_choice})
        
        # fixed slope for beta
        beta = pm.Normal("beta", mu = 0, sigma = {sigma_choice})
        
        # varying intercepts & slopes for alpha
        alpha_var = pm.Normal(
            "alpha_var", 
            mu = alpha, 
            sigma = sigma_alpha, 
            dims = "idx")
        
        # expected value per participant at each time-step
        mu = alpha_var[idx_] + beta * t_ 

        # model error
        sigma = pm.HalfNormal("sigma", sigma = {sigma_choice})
        
        # likelihood
        y_pred = pm.Normal(
            "y_pred", 
            mu = mu, 
            sigma = sigma, 
            observed = y_train, 
            dims = dims)
        
        # return model
        return {model_name}

# now run the function to compile the model
{model_name} = intercept()
'''
    return py_code


def py_covariation(model_name, sigma_choice):
    py_code = f'''
def covariation(): 
    with pm.Model(coords=coords) as {model_name}: 
            
        # Inputs
        idx_ = pm.Data('idx_shared', idx_train, dims = ('idx', 't'))
        t_ = pm.Data('t_shared', t_train, dims = ('idx', 't'))

        
        # prior stddev in intercepts & slopes (variation across counties):
        sd_dist = pm.HalfNormal.dist({sigma_choice}) # distribution. 

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
        alpha = pm.Normal("alpha", mu = 1.5, sigma = {sigma_choice})
        beta = pm.Normal("beta", mu = 0, sigma = {sigma_choice})
        
        # population of varying effects
        alpha_beta = pm.MvNormal(
            "alpha_beta", 
            mu = tt.stack([alpha, beta]), 
            chol = chol, 
            dims=("idx", "param"))

        # expected value per participant at each time-step
        mu = alpha_beta[idx_, 0] + alpha_beta[idx_, 1] * t_

        # model error
        sigma = pm.HalfNormal("sigma", sigma = {sigma_choice})
        
        # likelihood
        y_pred = pm.Normal(
            "y_pred", 
            mu = mu, 
            sigma = sigma, 
            observed = y_train, 
            dims = ('idx', 't'))
        
        # return the model
        return {model_name}

# now run the function to compile the model
{model_name} = covariation()
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
def py_pp(model_name, idata_name): 
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
### python: plot hdi (fixed effects) ###
# take out posterior predictive from idata
post_pred = {idata_name}.posterior_predictive

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
### python: plot hdi (full uncertainty) ###
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
### python: HDI for parameters ###
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
# if you do not already have the test data from simulation
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
lm = sns.lmplot(x = "t", y = "y", hue = "idx", data = train) # seaborn
lm.fig.suptitle("Python: Quick EDA") # add title
plt.plot(); # show plot
    '''
    return py_code 

def py_loo(): 
    py_code = f'''
### python: model comparison ###
loo_overview = az.compare({{
    "m_pooled": idata_pooled,
    "m_intercept": idata_intercept,
    "m_covariation": idata_covariation}})
    
loo_overview
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

def R_sim(): 
    R_code = f'''
### R: simulate data ###
set.seed(42)
d <- tibble(idx = seq(0, 14),
            t = seq(0, 14)) %>%
  data_grid(idx, t) %>%
  group_by(idx) %>%
  mutate(a_real = rnorm(1, 1, 0.5),
         b_real = rnorm(1, 0.3, 0.2),
         eps_real = rnorm(15, 0, 0.5),
         y = a_real + b_real * t + eps_real) %>%
  select(c(idx, t, y)) %>%
  mutate(idx = as_factor(idx))
  
# split into test & train
train <- d %>%
  filter(t <= 9)

test <- d %>%
  filter(t > 9)

# save data (we use python data in the app)
write_csv(train, "../data/train_R.csv")
write_csv(test, "../data/test_R.csv")
    '''
    return R_code

# r preprocessing
def R_preprocessing():
    R_code = f'''
### R: preprocessing ###
# we'll grab the training data from python for comparability
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
    prior(normal(0, {sigma_choice}), class = b), # beta
    prior(normal(1.5, {sigma_choice}), class = Intercept), # alpha
    prior(normal(0, {sigma_choice}), class = sigma)) # sigma (model error)

# compile model & sample prior
{model_name} <- brm(
    formula = {model_formula}, # model formula 
    family = gaussian, # likelihood function
    data = train,
    prior = {prior_name}, 
    sample_prior = "only", # only sample prior (for pp checks)
    backend = "cmdstanr", # faster than rstan
    seed = RANDOM_SEED) # set in "Packages & Reproducibility
'''
    return R_code

def R_covariation(model_name, model_formula, prior_name, sigma_choice):
    R_code = f'''
### R: specify model & compile ###
# formula 
{model_formula} <- bf(y ~ 1 + t + (1+t|idx)) # random intercepts, slopes & cov./corr.
    
# set priors --> can use get_prior() if in doubt. 
{prior_name} <- c(
    prior(normal(0, {sigma_choice}), class = b), # beta
    prior(normal(1.5, {sigma_choice}), class = Intercept), # alpha
    prior(normal(0, {sigma_choice}), class = sd), # sd
    prior(normal(0, {sigma_choice}), class = sigma), # sigma (model error)
    prior(lkj(2), class = cor) # covariation/corr. of random eff. 
)

# compile model & sample prior
{model_name} <- brm(
    formula = {model_formula}, # model formula
    family = gaussian, # likelihood function
    data = train,
    prior = {prior_name},
    sample_prior = "only", # only sample prior 
    backend = "cmdstanr") # faster than rstan
    '''
    return R_code

def R_intercept(model_name, model_formula, prior_name, sigma_choice):
    R_code = f'''
### R: specify model & compile ###
# formula 
{model_formula} <- bf(y ~ 1 + t + (1|idx)) # random intercepts

# set priors --> can use get_prior() if in doubt. 
{prior_name} <- c(
    prior(normal(0, {sigma_choice}), class = b), # beta
    prior(normal(1.5, {sigma_choice}), class = Intercept), # alpha
    prior(normal(0, {sigma_choice}), class = sd), # sd
    prior(normal(0, {sigma_choice}), class = sigma) # model error (sigma)
)

# compile model & sample prior
{model_name} <- brm(
    formula = {model_formula}, # model formula 
    family = gaussian, # likelihood function
    data = train,
    prior = {prior_name},
    sample_prior = "only", # only sample prior
    backend = "cmdstanr", # faster than rstan
    seed = RANDOM_SEED) # set in "Packages & Reproducibility
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
def R_sample(model_name, model_formula, model_family, prior_name, model_context, prior):
    R_code = f'''
### R: sample posterior ###
{model_name} <- brm(
    formula = {model_formula}, # model formula
    family = {model_family}, # likelihood function
    data = train,
    prior = {prior_name}, 
    sample_prior = TRUE, # sample prior and posterior 
    backend = "cmdstanr", # faster than rstan
    chains = 2,
    cores = 4, # difference between prior/posterior sampling
    iter = 4000, 
    warmup = 2000,
    threads = threading(2), # not sure this can be done in pyMC3
    control = list(adapt_delta = .99,
                    max_treedepth = 20),
    file = "../models_R/m_{model_context}_{prior}_fit", # name of saved model file
    file_refit = "on_change", # refit on change, otherwise just compile
    seed = RANDOM_SEED) # set in "Packages & Reproducibility" 
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
                    .width = c(.95, .8), # intervals
                    color = "#08519C") + 
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
    {function}({model_name}, re_formula = NA) %>%
    ggplot(aes(x = t, y = y)) + 
    stat_lineribbon(aes(y = {pred_type}), 
                    .width = c(.95, .8), # Intervals
                    color = "#08519C") + 
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
                    .width = c(.95, .8), # Intervals
                    color = "#08519C") + 
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
# if you do not have the test data from the simulation in your workspace
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

def R_loo(): 
    R_code = f'''
### R: run LOO comparison

# add criterion
m_pooled <- add_criterion(
    m_pooled, 
    criterion = c("loo", "bayes_R2"))

m_intercept <- add_criterion(
    m_intercept, 
    criterion = c("loo", "bayes_R2"))

m_covariation <- add_criterion(
    m_covariation, 
    criterion = c("loo", "bayes_R2"))

# run loo compare
loo_compare(m_pooled,
            m_intercept,
            m_covariation)
            
# model weights by stacking (as in pyMC3)
loo_model_weights(m_pooled,
                  m_intercept,
                  m_covariation)
    '''
    return R_code


