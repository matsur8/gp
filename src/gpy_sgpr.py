import argparse 

import numpy as np
import GPy

from gpy_basic import predict, show_model

def make_model(X, y, optimize, lengthscale, variance, noise_variance, n_inducing_inputs, fix_inducing_inputs):
    kernel = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
    model = GPy.models.SparseGPRegression(X, y[:,np.newaxis], Z=X[np.random.choice(X.shape[0], n_inducing_inputs),:], kernel=kernel)
    model.likelihood.variance = noise_variance
    if not optimize:
        model.kern.fix()
        model.likelihood.fix()
    if fix_inducing_inputs:
        model.Z.fix()
    if optimize or not fix_inducing_inputs:
        model.optimize()
    return model

def parse_make_model_option(s):
    parser = argparse.ArgumentParser()
    parser.add_argument("n_inducing_inputs", type=int)
    parser.add_argument("--fix_inducing_inputs", "-f", action="store_true")
    args = parser.parse_args(s.split())
    return vars(args)



