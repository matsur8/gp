import glob

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_list = glob.glob("results/*csv")


for var_col in ["MSE", "time_total", "MLL"]:
    for path in path_list:
        d = pd.read_csv(path)
        if not d["optimize"][0]:
            continue
        n_col = "n_train" if "n_train" in d.columns else "n"
        if var_col not in d.columns:
            continue
        n_max = d[n_col].max()
        X = np.c_[d[n_col]**3, d[n_col]**2, d[n_col], np.ones(d.shape[0])]
        y = d[var_col]
        beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
        #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
        print(beta)
        dd = d.groupby(n_col)[var_col].mean().reset_index()
        dd = dd[dd[n_col] >= 800]
        plt.semilogx(dd[n_col], dd[var_col], "o", label=d["module"][0] + str(d["make_model_option"][0]) +("_optimize" if d["optimize"][0] else ""))
        
    plt.legend(loc=1)
    plt.xlabel("number of training points")
    plt.ylabel(var_col)
    plt.tight_layout()
    plt.savefig(var_col + "_optimize.png")
    plt.close()


    for path in path_list:
        print(1, path)
        d = pd.read_csv(path)
        if d["optimize"][0]:
            continue
        n_col = "n_train" if "n_train" in d.columns else "n"
        if var_col not in d.columns:
            print(path)
            continue
        n_max = d[n_col].max()
        X = np.c_[d[n_col]**3, d[n_col]**2, d[n_col], np.ones(d.shape[0])]
        y = d[var_col]
        beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
        #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
        print(beta)
        dd = d.groupby(n_col)[var_col].mean().reset_index()
        dd = dd[dd[n_col] >= 800]
        plt.semilogx(dd[n_col], dd[var_col], "o", label=d["module"][0] + str(d["make_model_option"][0]) + ("_optimize" if d["optimize"][0] else ""))


    plt.legend(loc=1)
    plt.xlabel("number of training points")
    plt.ylabel(var_col)
    plt.tight_layout()
    plt.savefig(var_col + ".png")
    plt.close()


