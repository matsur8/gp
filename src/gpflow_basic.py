import gpflow
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def make_model(X, y, optimize, variance, lengthscale, noise_variance):
    k = gpflow.kernels.RBF(input_dim=1, variance=variance, lengthscales=lengthscale)
    m = gpflow.models.GPR(X, y[:,np.newaxis], kern=k)
    m.likelihood.variance = noise_variance
    if optimize:
        gpflow.train.ScipyOptimizer().minimize(m)
    return m

def predict(X, m):
    p =  m.predict_y(X)
    return p[0][:,0], np.sqrt(p[1][:,0])

def show_model(m):
    print(m.as_pandas_table())

