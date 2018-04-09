import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_list = glob.glob("results/*csv")


for path in path_list:
    d = pd.read_csv(path)
    n_col = "n_train" if "n_train" in d.columns else "n"
    var_col = "SMSE"
    if var_col not in d.columns:
        continue
    n_max = d[n_col].max()
    X = np.c_[d[n_col]**3, d[n_col]**2, d[n_col], np.ones(d.shape[0])]
    y = d[var_col]
    beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
    #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
    print(beta)
    plt.loglog(d[n_col], d[var_col], "o", label=d["module"][0] + ("_optimize" if d["optimize"][0] else ""))


plt.legend(loc=2)
plt.xlabel("number of training points")
plt.ylabel("time (s)")
plt.savefig("v.png")
plt.close()

