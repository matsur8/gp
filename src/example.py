import argparse
import importlib
import time

import numpy as np
import pandas as pd
import GPy

parser = argparse.ArgumentParser()
parser.add_argument("module")
parser.add_argument("--optimize", action="store_true")
parser.add_argument("--make_model_option")
parser.add_argument("--predict_option")

args = parser.parse_args()

np.random.seed(141)

m = importlib.import_module(args.module)

if args.make_model_option:
    make_model_option = m.parse_make_model_option(args.make_model_option)
else:
    make_model_option = {}

if args.predict_option:
    predict_option = m.parse_predict_option(args.predict_option)
else:
    predict_option = {}

#setting for benchmark
n_train = 200
n_test = 100
dim = 2

lengthscale_true = 3.0
variance_true = 2.0
noise_variance_true = 0.2

if args.optimize:
    lengthscale_init = 1.0
    variance_init = 1.0
    noise_variance_init = 1.0
else:
    lengthscale_init = lengthscale_true 
    variance_init = variance_true
    noise_variance_init = noise_variance_true

#sample data
rs = np.random.RandomState(22)
n = n_train + n_test
X = 10 * rs.uniform(size=(n,dim))#rs.random_sample((n, dim))
k = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale_true, variance=variance_true)
cov = k.K(X,X) 
f = rs.multivariate_normal(np.zeros(n), cov)
y = f + np.sqrt(noise_variance_true) * rs.normal(size=n)
X_train, X_test, y_train, y_test = X[:n_train], X[n_train:], y[:n_train], y[n_train:]


model = m.make_model(X_train, y_train, optimize=args.optimize, lengthscale=lengthscale_init, variance=variance_init, noise_variance=noise_variance_init, **make_model_option)

y_predict_mean, y_predict_std = m.predict(X_test, model, **predict_option)

MSE = ((y_test - y_predict_mean)**2).mean()
MLL = (0.5 * np.log(y_predict_std**2) + (y_test - y_predict_mean)**2 / (2*y_predict_std**2)).mean()
        
print(y_predict_mean[:5])
print(y_predict_std[:10])
print(MSE, MLL)
m.show_model(model)
print(m.get_hyp(model))

#import matplotlib.pyplot as plt
#plt.plot(X[:,0], f, "o")
#plt.plot(X[:,0], y, "o", color="red")
#plt.show()
#plt.plot(X_test[:,0], f[n_train:], "o")
#plt.plot(X_test[:,0], y_test, "o", color="red")
#plt.plot(X_test[:,0], y_predict_mean, "o", color="blue")
#plt.show()

#print(np.mean((f[n_train:] - y_test)**2))
