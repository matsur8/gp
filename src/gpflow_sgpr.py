import gpflow
import numpy as np

from gpflow_basic import predict, show_model

def make_model(X, y, optimize, lengthscale, variance, noise_variance, n_inducing_inputs):
    kernel = gpflow.kernels.RBF(input_dim=X.shape[1], lengthscales=lengthscale, variance=variance)
    model = gpflow.models.SGPR(X, y[:,np.newaxis], Z=X[np.random.choice(X.shape[0], n_inducing_inputs),:], kern=kernel)
    model.likelihood.variance = noise_variance
    if optimize:
        gpflow.train.ScipyOptimizer().minimize(model)
    return model

def parse_make_model_option(s):
    return {"n_inducing_inputs": int(s)}

