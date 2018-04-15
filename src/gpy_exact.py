import numpy as np
import GPy

def make_model(X, y, optimize, lengthscale, variance, noise_variance):
    kernel = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=lengthscale, variance=variance)
    model = GPy.models.GPRegression(X, y[:,np.newaxis], kernel)
    model.likelihood.variance = noise_variance
    if optimize:
        model.optimize()
    return model

def predict(X, model):
    p =  model.predict(X)
    return p[0][:,0], np.sqrt(p[1][:,0])

def show_model(model):
    print(model)

def get_hyp(model):
    return {"lengthscale": model.rbf.lengthscale.values[0],
            "variance": model.rbf.variance.values[0],
            "noise_variance": model.likelihood.variance.values[0]}


