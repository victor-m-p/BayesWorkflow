'''
This document is for things that are nice,
but not used at present.
'''

''' Dictionary way

##### first we fit a model ######
with pm.Model() as m0: 
    
    # shared variables 
    t_shared = pm.Data('t_shared', t)
    idx_shared = pm.Data('idx_shared', idx)
    
    # specify priors  
    beta = pm.Normal("beta", mu = 0, sigma = 0.3, shape = n_id)
    alpha = pm.Normal("alpha", mu = 0, sigma = 0.5, shape = n_id)
    eps = pm.HalfNormal("eps", sigma = 0.5) # this is explicitly half in pymc3

    # calculate mu
    mu = alpha[idx_shared] + beta[idx_shared] * t_shared
    
    # likelihood 
    y_pred = pm.Normal("y_pred", mu = mu, sigma = eps, observed = y)
    
    # sample and get prior and posterior predictive 
    trace = pm.sample(return_inferencedata = True)
    pprior = pm.sample_prior_predictive(samples = 50)
    ppost = pm.sample_posterior_predictive(trace, samples = 50)
    
    # gather it 
    m0_dct = {
        "model": m0,
        "trace": trace,
        "prior": pprior,
        "post": ppost}

## check the model ## 
az.plot_ppc(az.from_pymc3(
    posterior_predictive = m0_dct.get('ppost'), 
    model = m0_dct.get('model')));
    
'''

''' Posterior & Prior predictive
## m0 
with m0:
    m0_trace = pm.sample(return_inferencedata = True)
    ppost = pm.sample_posterior_predictive(m0_trace, samples = 50)
    pprior = pm.sample_prior_predictive(samples = 50)

## posterior predictive checks
az.plot_ppc(az.from_pymc3(
    posterior_predictive = ppost,
    model = m0
))

## prior predictive checks
az.plot_ppc(az.from_pymc3(
    posterior_predictive = pprior,
    model = m0
))

'''

''' prior/posterior
# https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html : structure

If you are working on models that are "slow"
you might want to sample the prior first.
That can be done as (below). 
?az.from_pymc3
## prior sampling 
with m0:
    prior_pred = pm.sample_prior_predictive(1000) # like setting this low. 
    m0_idata = az.from_pymc3(prior=prior_pred)
    
az.plot_ppc(m0_idata, group="prior"); # do we need "prior"

## posterior sampling 
with m0:
    post = pm.sample(return_inferencedata=True)
m0_idata.extend(post)

## posterior predictive 
with m0:
    post_pred = pm.sample_posterior_predictive(m0_idata, 1000)
    post_pred_idata = az.from_pymc3(posterior_predictive=post_pred)
m0_idata.extend(post_pred_idata)

az.plot_ppc(m0_idata, num_pp_samples = 100)
'''

'''

# Osvaldo Martin
_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
ax = np.ravel(ax)
j, k = 0, n_time
for i in range(n_id):
    ax[i].scatter(t[j:k], y[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', rotation=0, labelpad=15)
    #ax[i].set_xlim(6, 15)
    #ax[i].set_ylim(0, 17)
    j += n_time
    k += n_time
plt.tight_layout()

'''