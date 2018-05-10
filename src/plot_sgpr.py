import numpy as np
import pandas as pd
import GPy

import gpy_sgpr, gpy_exact
import matplotlib.pyplot as plt

np.random.seed(43141)

m = gpy_exact


n_train = 100 #100 or 1000
n_test = 100
dim = 1



noise_variance = 5.0 #5.0 or 0.1

#sample data
rs = np.random.RandomState(20)
n = n_train + n_test
X = rs.uniform(size=(n,dim)) 
#k = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale_true, variance=variance_true)
#cov = k.K(X,X) 
omega = 20 #20 or 200
f = np.cos(omega * X)[:,0]
y = f + np.sqrt(noise_variance) * rs.normal(size=n)
X_train, X_test, y_train, y_test = X[:n_train], X[n_train:], y[:n_train], y[n_train:]

lengthscale = 1.0 / omega
variance = 0.3

#model = gpy_sgpr.make_model(X_train, y_train, optimize=True, lengthscale=lengthscale, variance=variance, noise_variance=noise_variance, n_inducing_inputs=20, fix_inducing_inputs=False)
model = m.make_model(X_train, y_train, optimize=True, lengthscale=lengthscale, variance=variance, noise_variance=noise_variance)

y_predict_mean, y_predict_std = m.predict(X_test, model)

MSE = ((y_test - y_predict_mean)**2).mean()
MLL = (0.5 * np.log(y_predict_std**2) + (y_test - y_predict_mean)**2 / (2*y_predict_std**2)).mean()
        
print(y_predict_mean[:5])
print(y_predict_std[:10])
print(MSE, MLL)
m.show_model(model)
print(m.get_hyp(model))

p = model.plot()
p.plot(np.arange(0,1,0.01/omega), np.cos(omega*np.arange(0,1,0.01/omega)), "-", color="orange", label="true mean")
plt.legend()
plt.ylim(-10,10)
plt.show()

#import matplotlib.pyplot as plt
#plt.plot(X[:,0], f, "o")
#plt.plot(X[:,0], y, "o", color="red")
#plt.show()
#plt.plot(X_test[:,0], f[n_train:], "o")
#plt.plot(X_test[:,0], y_test, "o", color="red")
#plt.plot(X_test[:,0], y_predict_mean, "o", color="blue")
#plt.show()

