import argparse
import importlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import GPy


#setting for benchmark
n_train = 400
n_test = 100
dim = 1
T = 1 #number of experiments

lengthscale_true = 3.0
variance_true = 2.0
noise_variance_true = 0.2


rs = np.random.RandomState(3152)

#sample data
#Use rs to generate random-numbers.
n = n_train + n_test
X = 10 * rs.random_sample((n, dim))
k = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale_true, variance=variance_true)
cov = k.K(X,X) 
f = rs.multivariate_normal(np.zeros(n), cov)
y = f + np.sqrt(noise_variance_true) * rs.normal(size=n)
X_train, X_test, y_train, y_test = X[:n_train], X[n_train:], y[:n_train], y[n_train:]

print("s1")
time.sleep(2)
print("s2")


def make_rbf_feature(x, ls, v, delta, xgrid):
   return np.sqrt(delta * v) * (2.0 / (np.pi * ls**2))**0.25 * np.exp(-(xgrid - x)**2.0/ls**2.0)

l = -15
u = 25
delta = 0.01
xgrid = np.arange(-15, 20, delta)
n_grid = xgrid.shape[0]
w = rs.normal(size=n_grid)
f2 = np.array([make_rbf_feature(x[0], lengthscale_true, variance_true, delta, xgrid).dot(w) for x in X])

cov2 = np.zeros((n, n))
for i in range(n):
   fi = make_rbf_feature(X[i,0], lengthscale_true, variance_true, delta, xgrid)
   for j in range(n):
      fj = make_rbf_feature(X[j,0], lengthscale_true, variance_true, delta, xgrid)
      cov2[i,j] = fi.dot(fj)

plt.plot(X[:,0], f, "o")
plt.show()
plt.plot(X[:,0], f2, "o")
plt.show()

print(f.var())
print(f2.var())
print(np.max(np.abs(cov - cov2)))
print(np.max(np.abs(cov - cov2)/np.abs(cov)))
