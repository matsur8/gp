import glob

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

path_list = glob.glob("results/*csv")


for path in path_list:
    d = pd.read_csv(path)
    n_col = "n_train" if "n_train" in d.columns else "n"
    time_col = "time_total" if "time_total" in d.columns else "total"
    if d["optimize"][0]:
        n_max = d[n_col].max()
        X = np.c_[d[n_col]**3, d[n_col]**2, d[n_col], np.ones(d.shape[0])]
        y = d[time_col]
        beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
        #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
        print(beta)
        plt.loglog(d[n_col], d[time_col], "o", label=d["module"][0] + str(d["make_model_option"][0]) + ("_optimize" if d["optimize"][0] else ""))


plt.legend(loc=2)
plt.xlabel("number of training points")
plt.ylabel("time (s)")
plt.tight_layout()
plt.savefig("time_optimize.png")
plt.close()

for path in path_list:
    d = pd.read_csv(path)
    n_col = "n_train" if "n_train" in d.columns else "n"
    time_col = "time_total" if "time_total" in d.columns else "total"
    if not d["optimize"][0]:
        n_max = d[n_col].max()
        X = np.c_[d[n_col]**3, d[n_col]**2, d[n_col], np.ones(d.shape[0])]
        y = d[time_col]
        beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
        #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
        print(beta)
        plt.loglog(d[n_col], d[time_col], "o", label=d["module"][0] + str(d["make_model_option"][0]) + ("_optimize" if d["optimize"][0] else ""))


plt.legend(loc=2)
#, bbox_to_anchor=(0.5, -0.1))
plt.xlabel("number of training points")
plt.ylabel("time (s)")
plt.tight_layout()
plt.savefig("time.png")


