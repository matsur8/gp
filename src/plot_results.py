import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_list = glob.glob("results/*csv")

for path in path_list:
    d = pd.read_csv(path)
    print(d)
    n_max = d["n"].max()
    X = np.c_[d["n"]**3, d["n"]**2, d["n"], np.ones(d.shape[0])]
    y = d["total"]
    beta = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(y))
    #beta =  beta / ((np.ones(5)*n_max)**[4,3,2,1,0])
    print(beta)
    plt.loglog(d["n"], d["total"], "o-", label=d["module"][0] + ("_fit" if d["fit"][0] else ""))

plt.legend()
plt.show()

