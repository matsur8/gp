import argparse
import importlib
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("module")
parser.add_argument("--optimize", action="store_true")
args = parser.parse_args()

m = importlib.import_module(args.module)

np.random.seed(8)

n = 200
n_test = 100
X = 10 * np.random.random((n, 1))
y = np.sin(X)[:,0]
y += np.random.normal(size=X.shape[0])
X_test = 10 * np.random.random((n_test, 1))

model = m.make_model(X, y, optimize=args.optimize)
y_test_mean, y_test_std = m.predict(X_test, model)
print(y_test_mean[:5])
print(y_test_std[:5])
m.show_model(model)
