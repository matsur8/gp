import numpy as np
import GPy

def make_model(X, y, optimize):
    k = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
    m = GPy.models.GPRegression(X, y[:,np.newaxis], k)
    m.likelihood.variance = 0.1
    if optimize:
        m.optimize()
    return m

def predict(X, m):
    p =  m.predict_noiseless(X)
    return p[0][:,0], np.sqrt(p[1][:,0])





