'''
VMP: generate data
'''

## packages ##
import numpy as np
import pandas as pd
import pickle 
import seaborn as sns
import fun_helper as fh

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

## save data w. pickle.dump ##
with open('../data/train.pickle', 'wb') as f:
    pickle.dump(train, f)

with open("../data/test.pickle", 'wb') as f: 
    pickle.dump(test, f)

## save data for R
train.to_csv("../data/train.csv", index = False)
test.to_csv("../data/test.csv", index = False)
