import numpy as np
import GPy

def make_model(X, y, optimize=True, variance, lengthscale, noise_variance):
    kernel = GPy.kern.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)
    m = GPy.models.SparseGPRegression(X, y[:,np.newaxis], Z=X[np.random.choice(X.shape[0], 10),:].copy(), kernel=kernel)
    m.likelihood.variance = noise_variance
    if optimize:
        m.optimize()
    return m

def predict(X, m):
    r =  m.predict(X)
    return r[0][:,0], np.sqrt(r[1][:,0])

def show_model(m):
    print(m[""])




