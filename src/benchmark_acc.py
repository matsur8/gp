import argparse
import importlib
import time

import numpy as np
import pandas as pd
import GPy

parser = argparse.ArgumentParser()
parser.add_argument("module")
parser.add_argument("--optimize", action="store_true")
parser.add_argument("--limit_n_train", type=int, default=800)
args = parser.parse_args()

m = importlib.import_module(args.module)

np.random.seed(5)

n_train_list = [100*2**i for i in range(int(np.log2(args.limit_n_train/100)) + 1)]
#n_train_list = [10, 20]
T = 10
n_test = 2000

lengthscale_true = 2.0
variance_true = 0.8
noise_variance_true = 0.2

if args.optimize:
    lengthscale = 1.0
    variance = 1.0
    noise_variance = 1.0
else:
    lengthscale = lengthscale_true 
    variance = variance_true
    noise_variance = noise_variance_true

rs = np.random.RandomState(1252)

def sample_data(n_train, n_test, lengthscale, variance, noise_variance):
    n = n_train + n_test
    X = 10 * rs.random_sample((n, 1))
    k = GPy.kern.RBF(input_dim=1, lengthscale=lengthscale, variance=variance)
    cov = k.K(X,X) 
    y = rs.multivariate_normal(np.zeros(n), cov) + np.sqrt(noise_variance) * np.random.normal(size=n)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

res = []
for n_train in n_train_list:
    for t in range(T):
        X_train, X_test, y_train, y_test = sample_data(n_train, n_test, lengthscale_true, variance_true, noise_variance_true)
        s_make_model_time = time.time()
        model = m.make_model(X_train, y_train, args.optimize, lengthscale=lengthscale, variance=variance, noise_variance=noise_variance)
        e_make_model_time = time.time()
        s_predict_time = time.time()
        y_predict_mean, y_predict_std = m.predict(X_test, model)
        e_predict_time = time.time()
        #SMSE = ((y_test - y_predict_mean)**2).mean() / np.var(y_test)
        MSE = ((y_test - y_predict_mean)**2).mean()
        #MSLL = (0.5 * np.log(y_predict_std**2 / np.var(y_train)) + (y_test - y_predict_mean)**2 / (2*y_predict_std**2) - (y_test - y_train.mean())**2 / (2*np.var(y_train))).mean()
        MLL = (0.5 * np.log(y_predict_std**2) + (y_test - y_predict_mean)**2 / (2*y_predict_std**2)).mean()
        res.append((n_train, n_test, t, MSE, MLL, e_make_model_time - s_make_model_time, e_predict_time - s_predict_time))
        #print(np.c_[y_test, y_predict_mean, np.ones(n_test)*np.mean(y_train)])

#res = np.array(res)

res = pd.DataFrame(res)
res.columns = ["n_train", "n_test", "t", "SMSE", "MSLL", "time_make_model", "time_predict"]
res["time_total"] = res["time_make_model"] + res["time_predict"]
res.insert(0, "optimize", args.optimize)
res.insert(0, "module", args.module)

res.to_csv("results/" + args.module + ("_optimize" if args.optimize else "") + ".csv", index=False)
