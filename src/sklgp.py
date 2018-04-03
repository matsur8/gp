import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def make_model(X, y, optimize, variance, lengthscale, noise_variance):
    kernel = variance * RBF(length_scale=lengthscale) + WhiteKernel(noise_level=noise_variance)
    if optimize:
        # default optimizer = "fmin_l_bfgs_b"
        m = GaussianProcessRegressor(kernel=kernel,
                                     alpha=0.0
                                     #copy_X_train=False
                                 )
    else:
        m = GaussianProcessRegressor(kernel=kernel,
                                     alpha=0.0,
                                     optimizer=None
                                     #copy_X_train=False
                                 )
    m.fit(X, y)
    return m

def predict(X, m):
    p = m.predict(X, return_std=True)
    return p[0], np.sqrt(p[1]**2 + m.get_params()["alpha"])

def show_model(m):
    print(m.kernel_)

