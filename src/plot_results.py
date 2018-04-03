import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_list = glob.glob("results/time_*csv")


for path in path_list:
    d = pd.read_csv(path)
    if d["optimize"][0]:
        n_max = d["n"].max()
        X = np.c_[d["n"]**3, d["n"]**2, d["n"], np.ones(d.shape[0])]
        y = d["total"]
        beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
        #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
        print(beta)
        plt.loglog(d["n"], d["total"], "o-", label=d["module"][0] + ("_optimize" if d["optimize"][0] else ""))


plt.legend(loc=2)
plt.xlabel("number of training points")
plt.ylabel("time (s)")
plt.savefig("time_optimize.png")
plt.close()

for path in path_list:
    d = pd.read_csv(path)
    if not d["optimize"][0]:
        n_max = d["n"].max()
        X = np.c_[d["n"]**3, d["n"]**2, d["n"], np.ones(d.shape[0])]
        y = d["total"]
        beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
        #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
        print(beta)
        plt.plot(d["n"], d["total"], "o-", label=d["module"][0] + ("_optimize" if d["optimize"][0] else ""))


plt.legend(loc=2)
#, bbox_to_anchor=(0.5, -0.1))
plt.xlabel("number of training points")
plt.ylabel("time (s)")
plt.savefig("time.png")


