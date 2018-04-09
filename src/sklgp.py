import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def make_model(X, y, optimize, lengthscale, variance, noise_variance):
    kernel = variance * RBF(length_scale=lengthscale) + WhiteKernel(noise_level=noise_variance)
    if optimize:
        # default optimizer = "fmin_l_bfgs_b"
        model = GaussianProcessRegressor(kernel=kernel,
                                     alpha=0.0
                                 )
    else:
        model = GaussianProcessRegressor(kernel=kernel,
                                     alpha=0.0,
                                     optimizer=None
                                 )
    model.fit(X, y)
    return model

def predict(X, model):
    p = model.predict(X, return_std=True)
    return p[0], np.sqrt(p[1]**2 + model.get_params()["alpha"])

def show_model(model):
    print(model.kernel_)

def get_hyp(model):
    return {"variance": model.kernel_.k1.k1.constant_value,
            "lengthscale": model.kernel_.k1.k2.length_scale,
            "noise_variance": model.kernel_.k2.noise_level}
