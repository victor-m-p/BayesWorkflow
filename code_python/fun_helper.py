'''
19-5-21 (VMP).
functions for fitting the models &
for preprocessing. 
'''

# packages
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az
import dataframe_image as dfi

# train test split
def train_test(d, split_column, train_size = .75):

    # pretty manual approach
    sort_val = np.sort(np.unique(d[split_column].values))
    min_val = min(sort_val)
    length = int(round(len(sort_val)*train_size, 0))
    train = d[d[split_column] <= min_val+length]
    test = d[d[split_column] > min_val+length]
    return train, test

# sampling models
def sample_mod(
    model, 
    posterior_draws = 2000, 
    post_pred_draws = 4000,
    prior_pred_draws = 4000):
    
    with model: 
        trace = pm.sample(
            return_inferencedata = False, 
            draws = posterior_draws,
            target_accept = .99,
            max_treedepth = 20) # tuning!
        post_pred = pm.sample_posterior_predictive(trace, samples = post_pred_draws)
        prior_pred = pm.sample_prior_predictive(samples = prior_pred_draws)
        m_idata = az.from_pymc3(trace = trace, posterior_predictive=post_pred, prior=prior_pred)
    
    return m_idata

# kruschke/plate diagram
def plot_plate(compiled_model, model_type): 
    g = pm.model_to_graphviz(compiled_model) 
    g.render(
        f"../plots_python/{model_type}_plate",
        format = "png")

# prior predictive
def prior_pred(m_idata, model_type, prior_level, n_draws = 100): 
    fig, ax = plt.subplots()
    az.plot_ppc(m_idata, group = "prior", num_pp_samples = n_draws, ax = ax)
    fig.suptitle("Python/pyMC3: prior predictive check")
    fig.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(10, 7)
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_prior_pred.jpeg",
                dpi = 300)

# posterior predictive
def posterior_pred(m_idata, model_type, prior_level, n_draws):
    fig, ax = plt.subplots()
    az.plot_ppc(m_idata, num_pp_samples = n_draws, ax = ax)
    fig.suptitle("Python/pyMC3: posterior predictive check")
    fig.tight_layout()
    figure = plt.gcf()
    figure.set_size_inches(10, 7)
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_posterior_pred.jpeg",
                dpi = 300)

# plot trace
def plot_trace(m_idata, model_type, prior_level):
    #fig, ax = plt.subplots()
    az.plot_trace(m_idata)
    #fig.suptitle("Python/pyMC3: trace plot")
    #fig.tight_layout() 
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_plot_trace.jpeg")
    
# export summary
def export_summary(m_idata, model_type, prior_level):
    summary = az.summary(m_idata)
    dfi.export(
        summary,
        f"../plots_python/{model_type}_{prior_level}_summary.png"
    )    

# updating checks
def updating_check(m_idata, n_prior = 100, n_posterior = 100): 
    fig, axes = plt.subplots(nrows = 2)

    az.plot_ppc(m_idata, group = "prior", num_pp_samples = n_prior, ax = axes[0])
    az.plot_ppc(m_idata, num_pp_samples = n_posterior, ax = axes[1])
    plt.draw()
