import numpy as np
import GPy

from gpy_basic import predict, show_model

def make_model(X, y, optimize, lengthscale, variance, noise_variance, n_inducing_inputs):
    kernel = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
    model = GPy.models.SparseGPRegression(X, y[:,np.newaxis], Z=X[np.random.choice(X.shape[0], n_inducing_inputs),:], kernel=kernel)
    model.likelihood.variance = noise_variance
    if optimize:
        model.optimize()
    return model

def parse_make_model_option(s):
    return {"n_inducing_inputs": int(s)}



