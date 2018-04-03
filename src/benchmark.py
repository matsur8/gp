import argparse
import importlib
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("module")
parser.add_argument("--optimize", action="store_true")
parser.add_argument("--limit_n_train", type=int,default=800)
args = parser.parse_args()

m = importlib.import_module(args.module)

np.random.seed(8)

n_list = [100*2**i for i in range(int(np.log2(args.limit_n_train/100)) + 1)]
#n_list = [10, 20]
n_test = 100
X = 10 * np.random.random((np.max(n_list), 1))
y = np.sin(X)[:,0]
y += np.random.normal(size=X.shape[0])
X_test = 10 * np.random.random((n_test, 1))

r = []
for n in n_list:
    s_make_model_time = time.time()
    model = m.make_model(X[:n,:], y[:n], optimize=args.optimize)
    e_make_model_time = time.time()
    s_predict_time = time.time()
    y_predict_mean, y_predict_std = m.predict(X_test[:n,:], model)
    e_predict_time = time.time()
        
    r.append((n, e_make_model_time - s_make_model_time, e_predict_time - s_predict_time))

print(y_predict_mean[:5], y_predict_std[:5])
r = np.array(r)

d = pd.DataFrame({"module": [args.module] * r.shape[0],
                  "optimize": [args.optimize] * r.shape[0],
                  "n": r[:,0].astype(np.int),
                  "make_model": r[:,1],
                  "predict": r[:,2],
                  "total": r[:,1] + r[:,2]},
                 columns = ["module", "optimize", "n", "make_model", "predict", "total"])

d.to_csv("results/" + args.module + ("_optimize" if args.optimize else "") + ".csv", index=False)
