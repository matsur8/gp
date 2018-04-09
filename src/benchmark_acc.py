import argparse
import datetime
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
parser.add_argument("--limit_n_train", type=int, default=800)

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
n_train_list = [100*2**i for i in range(int(np.log2(args.limit_n_train/100)) + 1)]
n_test = 1000
dim = 1
T = 1 #number of experiments

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

rs = np.random.RandomState(1252)

res = []
for n_train in n_train_list:
    for t in range(T):

        #sample data
        #Use rs to generate random-numbers.
        n = n_train + n_test
        X = 10 * rs.random_sample((n, dim))
        k = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale_true, variance=variance_true)
        cov = k.K(X,X) 
        y = rs.multivariate_normal(np.zeros(n), cov) + np.sqrt(noise_variance_true) * rs.normal(size=n)
        X_train, X_test, y_train, y_test = X[:n_train], X[n_train:], y[:n_train], y[n_train:]

        #make model
        s_make_model_time = time.time()
        model = m.make_model(X_train, y_train, optimize=args.optimize, lengthscale=lengthscale_init, variance=variance_init, noise_variance=noise_variance_init, **make_model_option)
        e_make_model_time = time.time()

        #predict
        s_predict_time = time.time()
        y_predict_mean, y_predict_std = m.predict(X_test, model, **predict_option)
        e_predict_time = time.time()

        #evaluate
        #SMSE = ((y_test - y_predict_mean)**2).mean() / np.var(y_test)
        MSE = ((y_test - y_predict_mean)**2).mean()
        #MSLL = (0.5 * np.log(y_predict_std**2 / np.var(y_train)) + (y_test - y_predict_mean)**2 / (2*y_predict_std**2) - (y_test - y_train.mean())**2 / (2*np.var(y_train))).mean()
        MLL = (0.5 * np.log(y_predict_std**2) + (y_test - y_predict_mean)**2 / (2*y_predict_std**2)).mean()
        
        hyp = m.get_hyp(model)
        res.append((n_train, n_test, t, MSE, MLL, e_make_model_time - s_make_model_time, e_predict_time - s_predict_time, 
                    e_make_model_time - s_make_model_time +  e_predict_time - s_predict_time,
                    hyp["lengthscale"], hyp["variance"], hyp["noise_variance"]))


res = pd.DataFrame(res)
res.columns = ["n_train", "n_test", "t", "MSE", "MLL", "time_make_model", "time_predict",
               "time_total", "lengthscale", "variance", "noise_variance"]
res.insert(0, "noise_variance_true", noise_variance_true)
res.insert(0, "variance_true", variance_true)
res.insert(0, "lengthscale_true", lengthscale_true)
res.insert(0, "predict_option", args.predict_option)
res.insert(0, "make_model_option", args.make_model_option)
res.insert(0, "optimize", args.optimize)
res.insert(0, "module", args.module)

now = datetime.datetime.now()

res.to_csv("results/"
    + args.module
    + ("_optimize" if args.optimize else "")
    + ("_" + args.make_model_option.replace(" ", "+") if args.make_model_option is not None else "")
    + ("_" + args.predict_option.replace(" ", "+") if args.predict_option is not None else "")
    + "_{:%Y%m%d%H%M%S}".format(now)
    + ".csv",
    index=False)
