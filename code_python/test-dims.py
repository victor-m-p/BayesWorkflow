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

#collapse-hide
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
srrs_mn["county_code"], mn_counties = pd.factorize(srrs_mn.county)
srrs_mn["log_radon"] = np.log(srrs_mn.activity + 0.1)

coords = {
    "Level": ["Basement", "Floor"], 
    "obs_id": np.arange(n),
    "County": mn_counties,
}

with pm.Model(coords=coords) as hierarchical_intercept:
    floor_idx = pm.Data("floor_idx", srrs_mn.floor, dims="obs_id")
    county_idx = pm.Data("county_idx", srrs_mn.county_code, dims="obs_id")
    uranium = pm.Data("uranium", u, dims="County")

    # Hyperpriors:
    g = pm.Normal("g", mu=0.0, sigma=10.0, shape=2)
    sigma_a = pm.Exponential("sigma_a", 1.0)

    # Varying intercepts uranium model:
    a = pm.Deterministic("a", g[0] + g[1] * uranium, dims="County")
    za_county = pm.Normal("za_county", mu=0.0, sigma=1.0, dims="County")
    a_county = pm.Deterministic("a_county", a + za_county * sigma_a, dims="County")
    # Common slope:
    b = pm.Normal("b", mu=0.0, sigma=1.0)

    # Expected value per county:
    theta = a_county[county_idx] + b * floor_idx
    # Model error:
    sigma = pm.Exponential("sigma", 1.0)

    y = pm.Normal("y", theta, sigma=sigma, observed=srrs_mn.log_radon.values, dims="obs_id")

    hierarchical_intercept_idata = pm.sample(
        2000, tune=2000, target_accept=0.99, random_seed=32, return_inferencedata=True
    )
    
hierarchical_intercept_idata

uranium = hierarchical_intercept_idata.constant_data.uranium
post = hierarchical_intercept_idata.posterior.assign_coords(uranium=uranium)
avg_a = post["a"].mean(dim=("chain", "draw")).sortby("uranium")
avg_a_county = post["a_county"].mean(dim=("chain", "draw"))
avg_a_county_hdi = az.hdi(post, var_names="a_county")["a_county"]

_, ax = plt.subplots()
ax.plot(avg_a.uranium, avg_a, "k--", alpha=0.6, label="Mean intercept")
az.plot_hdi(
    uranium,
    post["a"],
    fill_kwargs={"alpha": 0.1, "color": "k", "label": "Mean intercept HPD"},
    ax=ax,
)
ax.scatter(uranium, avg_a_county, alpha=0.8, label="Mean county-intercept")
ax.vlines(
    uranium,
    avg_a_county_hdi.sel(hdi="lower"),
    avg_a_county_hdi.sel(hdi="higher"),
    alpha=0.5,
    color="orange",
)
plt.xlabel("County-level uranium")
plt.ylabel("Intercept estimate")
plt.legend(fontsize=9);

## question then is... can we add more stuff to it?


