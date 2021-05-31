## packages 
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
import arviz as az
import sklearn as sk
import pickle
import fun_models as fm

## load data 
with open('../data/train.pickle', 'rb') as f:
    train = pickle.load(f)

with open('../data/test.pickle', 'rb') as f:
    test = pickle.load(f)

## exploratory plot
sns.lmplot(x = "t", y = "y", hue = "idx", data = train)
figure = plt.gcf()
figure.set_size_inches(10, 7)
plt.savefig("../plots_python/EDA.jpeg", dpi=300, transparent=False)
