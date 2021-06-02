## sources
# https://oriolabril.github.io/oriol_unraveled/python/arviz/pymc3/xarray/2020/09/22/pymc3-arviz.html
# https://docs.pymc.io/notebooks/multilevel_modeling.html?highlight=sampler

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

# Import radon data
srrs2 = pd.read_csv(pm.get_data("srrs2.dat"))
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state == "MN"].copy()

srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
cty = pd.read_csv(pm.get_data("cty.dat"))
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
u = np.log(srrs_mn.Uppm).unique()

n = len(srrs_mn)

srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(counties)))

county = srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
floor = srrs_mn.floor.values

coords = {
    "Level": ["Basement", "Floor"], 
    "obs_id": np.arange(floor.size),
    "County": mn_counties}

with pm.Model(coords=coords) as varying_intercept:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")
    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=10.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts:
    a_county = pm.Normal("a_county", mu=a, sigma=sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=10.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")
pm.model_to_graphviz(varying_intercept)

with varying_intercept:
    varying_intercept_idata = pm.sample(
        tune=2000, init="adapt_diag", random_seed=32, return_inferencedata=True
    )

import xarray as xr
xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
post = varying_intercept_idata.posterior  # alias for readability

# taking into account random effects.
theta = (
    (post.a_county + post.b * xvals).mean(dim=("chain", "draw")).to_dataset(name="Mean log radon")
)

test = theta["Mean log radon"].values

az.plot_hdi(
    [0, 1],
    test
)




## random slopes
with pm.Model(coords=coords) as varying_intercept_slope:
    floor_idx = pm.Data("floor_idx", floor, dims="obs_id")
    county_idx = pm.Data("county_idx", county, dims="obs_id")

    # Hyperpriors:
    a = pm.Normal("a", mu=0.0, sigma=5.0)
    sigma_a = pm.Exponential("sigma_a", 1.0)
    b = pm.Normal("b", mu=0.0, sigma=1.0)
    sigma_b = pm.Exponential("sigma_b", 0.5)

    # Varying intercepts:
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    # Varying slopes:
    zb_county = pm.Normal("zb_county", mu=0.0, sigma=1.0, dims="County")

    # Expected value per county:
    theta = (a + za_county[county_idx] * sigma_a) + (b + zb_county[county_idx] * sigma_b) * floor
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=log_radon, dims="obs_id")

    varying_intercept_slope_idata = pm.sample(
        2000, tune=2000, target_accept=0.99, random_seed=32, return_inferencedata=True
    )

# should not mean over draw..
xvals = xr.DataArray([0, 1], dims="Level", coords={"Level": ["Basement", "Floor"]})
post = varying_intercept_slope_idata.posterior  # alias for readability
avg_a_county = (post.a + post.za_county * post.sigma_a).mean(dim=("chain", "draw"))
avg_b_county = (post.b + post.zb_county * post.sigma_b).mean(dim=("chain", "draw"))
theta = (avg_a_county + avg_b_county * xvals).to_dataset(name="Mean log radon")
theta = theta["Mean log radon"].values
theta
post
az.plot_hdi(
    [0, 1],
    theta
)

_, ax = plt.subplots()
theta.plot.scatter(x="Level", y="Mean log radon", alpha=0.2, color="k", ax=ax)  # scatter
ax.plot(xvals, theta["Mean log radon"].T, "k-", alpha=0.2)
# add lines too
ax.set_title("MEAN LOG RADON BY COUNTY");

#### https://docs.pymc.io/notebooks/posterior_predictive.html
with varying_intercept_slope: 
    ppc = pm.sample_posterior_predictive(
        varying_intercept_slope_idata, var_names=["y", "a", "b"])
    idata_aux = az.from_pymc3(posterior_predictive=ppc)

# add it
varying_intercept_slope_idata.extend(idata_aux)
varying_intercept_slope_idata.posterior_predictive

# calculate mean outcome. 
mu_pp = (ppc["a"] + ppc["b"] * np.array([0, 1])[:, None]).T
mu_pp.shape

# plot data
fig, ax = plt.subplots()
ax.plot(floor, log_radon, 'o')

# plot mean outcome
ax.plot([0, 1], mu_pp.mean(0), label="Mean outcome", alpha=0.6)

# plot mean 94% HDI outcome
az.plot_hdi(
    [0, 1],
    mu_pp,
    ax=ax,
    fill_kwargs={"alpha": 0.3, "label": "Mean outcome 94% HPD"}
)

floor
ppc["y"].shape
# plot predictive 
az.plot_hdi(
    [0, 1],
    
)