from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def make_model(X, y, optimize):
    kernel = RBF(length_scale=1.0)
    if optimize:
        # default optimizer = "fmin_l_bfgs_b"
        m = GaussianProcessRegressor(kernel=kernel,
                                     alpha=0.1,
                                     copy_X_train=False)
    else:
        m = GaussianProcessRegressor(kernel=kernel,
                                     alpha=0.1,
                                     optimizer=None,
                                     copy_X_train=False)
    m.fit(X, y)
    return m

def predict(X, m):
    return m.predict(X, return_std=True)


