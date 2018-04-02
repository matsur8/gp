import argparse
import importlib
from resource import getrusage, RUSAGE_SELF
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("module")
parser.add_argument("--fit", action="store_true")
args = parser.parse_args()

m = importlib.import_module(args.module)

np.random.seed(8)

n_list = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
#n_list = [10, 20]
n_test = 100
X = 10 * np.random.random((np.max(n_list), 1))
y = np.sin(X)[:,0]
y += np.random.normal(size=X.shape[0])
X_test = 10 * np.random.random((n_test, 1))

r = []
for n in n_list:
    s_make_model_time = getrusage(RUSAGE_SELF)
    model = m.make_model(X[:n,:], y[:n], fit=args.fit)
    e_make_model_time = getrusage(RUSAGE_SELF)
    s_predict_time = getrusage(RUSAGE_SELF)
    y_pred_mean, y_pred_std = m.predict(X_test[:n,:], model)
    e_predict_time = getrusage(RUSAGE_SELF)
        
    r.append((n, 
              e_make_model_time.ru_utime - s_make_model_time.ru_utime,
              e_predict_time.ru_utime - s_predict_time.ru_utime,))

print(y_test_mean[:5], y_test_std[:5])
r = np.array(r)

d = pd.DataFrame({"module": [args.module] * r.shape[0],
                  "fit": [args.fit] * r.shape[0],
                  "n": r[:,0].astype(np.int),
                  "make_model": r[:,1],
                  "predict": r[:,2],
                  "total": r[:,1] + r[:,2]},
                 columns = ["module", "fit", "n", "make_model", "predict", "total"])

d.to_csv("results/" + args.module + ("_fit" if args.fit else "") + ".csv", index=False)
