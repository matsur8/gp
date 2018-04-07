import numpy as np
from oct2py import octave

octave.addpath("src")
octave.gpml_setup()

def make_model(X, y, optimize, lengthscale, variance, noise_variance):
    if optimize:
        hyp_gpml = octave.gpml_basic_make_model(hyp_gpml, X, y[:,np.newaxis])
        hyp = {"lengthscale": np.exp(hyp_gpml["cov"][0,0]),
               "variance": np.exp(hyp_gpml["cov"][1,0] * 2),
               "noise_variance": np.exp(hyp_gpml["lik"] * 2)}
    else:
        hyp = {"lengthscale": lengthscale,
               "variance": variance,
               "noise_variance": noise_variance}
    return (hyp, X, y)

def predict(X, model):
    hyp, X_train, y_train = model
    hyp_gpml = {"cov": np.array([np.log(hyp["lengthscale"]), 
                                 0.5 * np.log(hyp["variance"])])[:,np.newaxis],
                "mean": [],
                "lik": 0.5 * np.log(hyp["noise_variance"])}
    ym, ys2 = octave.gpml_basic_predict(hyp_gpml, X_train, y_train[:,np.newaxis], X, nout=2)
    return ym[:,0], np.sqrt(ys2[:,0])

def show_model(model):
    print(model[0])

