import gpflow
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def make_model(X, y, optimize):
    k = gpflow.kernels.RBF(input_dim=1, variance=1.0, lengthscales=1.0)
    m = gpflow.gpr.GPR(X, y[:,np.newaxis], kern=k)
    m.likelihood.variance = 0.1

    if optimize:
        m.optimize()

    return m

def predict(X, m):
    p =  m.predict_f(X)
    return p[0][:,0], np.sqrt(p[1][:,0])


