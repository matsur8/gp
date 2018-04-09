import numpy as np
from oct2py import octave

from gpml_basic import show_model, get_hyp

octave.addpath("src")
octave.gpml_setup()

def make_model(X, y, optimize, lengthscale, variance, noise_variance, n_grid):
    if optimize:
        hyp_gpml = {"cov": np.array([np.log(lengthscale), 
                                     0.5 * np.log(variance)] * X.shape[1])[:,np.newaxis],
                    "mean": [],
                    "lik": 0.5 * np.log(noise_variance)}
        hyp_gpml = octave.gpml_msgp_make_model_wp(hyp_gpml, X, y[:,np.newaxis], n_grid)
        return hyp_gpml
        hyp = {"lengthscale": np.exp(hyp_gpml["cov"][0,0]),
               "variance": np.exp(hyp_gpml["cov"][1,0] * 2),
               "noise_variance": np.exp(hyp_gpml["lik"] * 2),
               "proj":hyp_gpml["proj"]}
    else:
        hyp = {"lengthscale": lengthscale,
               "variance": variance,
               "noise_variance": noise_variance}
    return (hyp, X, y)

def predict(X, model, n_grid):
    hyp, X_train, y_train = model
    hyp_gpml = {"cov": np.array([np.log(hyp["lengthscale"]), 
                                 0.5 * np.log(hyp["variance"])] * X.shape[1])[:,np.newaxis],
                "mean": [],
                "lik": 0.5 * np.log(hyp["noise_variance"])}
    ym, ys2 = octave.gpml_msgp_predict(hyp_gpml, X_train, y_train[:,np.newaxis], X, n_grid, nout=2)
    return ym[:,0], np.sqrt(ys2[:,0])

def show_model(model):
    print(model[0])

def get_hyp(model):
    return model[0]

def parse_make_model_option(s):
    return {"n_grid": int(s)}

def parse_predict_option(s):
    return {"n_grid": int(s)}
