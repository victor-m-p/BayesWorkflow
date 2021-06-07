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
    az.plot_trace(m_idata, figsize = (4.8, 4.8)) # for R compatibility
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_plot_trace.jpeg")
    
# export summary
def export_summary(m_idata, model_type, prior_level):
    summary = az.summary(m_idata)
    dfi.export(
        summary,
        f"../plots_python/{model_type}_{prior_level}_summary.png"
    )    

# plot hdi & save
def plot_hdi(t, y, n_idx, m_idata, model_type, prior_level, kind = "all", hdi_prob = (.95, .8)):
    
    # take out ppc 
    if kind == "full" or kind == "fixed":
        ppc = m_idata.posterior_predictive
    elif kind == "predictions": 
        ppc = m_idata.predictions
    
    # unpack tuple & get unique t. 
    high, low = hdi_prob
    t_unique = np.unique(t)
    n_time = len(t_unique)
    
    # take out data from ppc.
    y_pred = ppc["y_pred"].mean(axis = 0).values
    y_mean = y_pred.mean(axis = (0, 1))

    # here the plots differ
    if kind == "full" or kind == "predictions": 
        outcome = y_pred.reshape((4000*n_idx, n_time)) # 4000 = 2000 (draws) * 2 (chains)
    elif kind == "fixed": 
        outcome = (ppc.alpha.values + ppc.beta.values * t_unique[:, None]).T
    
    # set-up plot
    fig, ax = plt.subplots(figsize = (10, 7))  

    # plot data
    ax.scatter(
        t, 
        y,
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
    
    # add legend, title and formatting. 
    ax.legend()
    fig.suptitle(f"Python/pyMC3: Prediction Intervals ({kind})")
    fig.tight_layout()
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_HDI_{kind}.jpeg",
                dpi = 300)

# HDI for parameters
def hdi_param(m_idata, model_type, prior_level):
    
    fig, ax = plt.subplots(figsize = (10, 7))
    
    az.plot_forest(
        m_idata,
        var_names=["alpha", "beta", "sigma"], 
        combined=True, # combine chains 
        kind='ridgeplot', # instead of default which does not show distribution
        ridgeplot_truncate=False, # do show the tails 
        hdi_prob = .8, # hdi prob .8 here. 
        ridgeplot_alpha = 0.5, # looks prettier
        ridgeplot_quantiles = [0.5], # show mean
        ax = ax # add to our axis
        )

    fig.suptitle("Python/pyMC3: HDI intervals for parameters")
    fig.tight_layout()
    
    plt.savefig(f"../plots_python/{model_type}_{prior_level}_HDI_param.jpeg",
                dpi = 300)
