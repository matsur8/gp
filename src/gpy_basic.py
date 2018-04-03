import numpy as np
import GPy

def make_model(X, y, optimize, variance, lengthscale, noise_variance):
    k = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    m = GPy.models.GPRegression(X, y[:,np.newaxis], k)
    m.likelihood.variance = noise_variance
    if optimize:
        m.optimize()
    return m

def predict(X, m):
    p =  m.predict_noiseless(X)
    return p[0][:,0], np.sqrt(p[1][:,0])

def show_model(m):
    print(m[""])



