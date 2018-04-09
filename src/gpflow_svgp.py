#reference
#http://gpflow.readthedocs.io/en/latest/notebooks/svi_test.html

import argparse 

import gpflow
import numpy as np

from gpflow_basic import predict, show_model

def make_model(X, y, optimize, lengthscale, variance, noise_variance, n_inducing_inputs, fix_inducing_inputs)):
    kernel = gpflow.kernels.RBF(input_dim=X.shape[1], lengthscales=lengthscale, variance=variance)
    model = gpflow.models.SVGP(X, y[:,np.newaxis], Z=X[np.random.choice(X.shape[0], n_inducing_inputs),:], 
                               kern=kernel, likelihood=gpflow.likelihoods.Gaussian(noise_variance), minibatch_size=100)
    if not optimize:
        model.kern.trainable = False
        model.likelihood.trainable = False
    if fix_inducing_inputs:
        model.feature.Z.trainable = False
    if optimize or not fix_inducing_inputs:
        opt = gpflow.train.AdagradOptimizer(learning_rate=0.1)
        opt.minimize(model)
    return model

def parse_make_model_option(s):
    parser = argparse.ArgumentParser()
    parser.add_argument("n_inducing_inputs", type=int)
    parser.add_argument("--fix_inducing_inputs", "-f", action="store_true")
    args = parser.parse_args(s.split())
    return vars(args)

