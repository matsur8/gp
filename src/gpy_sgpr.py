import numpy as np
import GPy

def make_model(X, y, optimize=True):
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=1.0)
    m = GPy.models.SparseGPRegression(X, y[:,np.newaxis], Z=X[np.random.choice(X.shape[0], 10),:].copy(), kernel=kernel)
    m.likelihood.variance = 0.1
    if optimize:
        m.optimize()
    return m

def predict(X, m):
    r =  m.predict_noiseless(X)
    return r[0][:,0], np.sqrt(r[1][:,0])





