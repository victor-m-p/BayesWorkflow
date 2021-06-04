'''
VMP: generate data
'''

## packages ##
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## simulate data ##
np.random.seed(17) # reproducibility
n_id = 15 # 8 people
n_time = 15 # 15 time-steps 
idx = np.repeat(range(n_id), n_time)
a_real = np.random.normal(loc = 1, scale = 0.5, size = n_id) # intercept
b_real = np.random.normal(loc = 0.3, scale = 0.2, size = n_id) # beta
eps_real = np.random.normal(loc = 0, scale = 0.5, size = len(idx)) # error
t = np.resize(np.arange(n_time), n_time*n_id)
y = a_real[idx] + b_real[idx] * t + eps_real # outcome

## data frame ##
d = pd.DataFrame({
    'idx': idx, 
    't': t,
    'y': y})

## train and test split
train, test = fh.train_test(d, "t", 0.6)

## quick EDA (title?)
sns.lmplot(x = "t", y = "y", hue = "idx", data = train)
figure = plt.gcf()
figure.set_size_inches(10, 7)
figure.title("Python: Quick EDA")
plt.savefig("../plots_python/EDA.jpeg", dpi=300, transparent=False)

## save data csv (for R compatibility)
train.to_csv("../data/train.csv", index = False)
test.to_csv("../data/test.csv", index = False)
