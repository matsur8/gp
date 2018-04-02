import numpy as np
import GPy

N = 1000
X = np.random.uniform(-3.,3.,(N,1))
Y = np.sin(X) + np.random.randn(N,1)*0.05

kernel = GPy.kern.RBF(input_dim=1, lengthscale=1.0)

m = GPy.models.GPRegression(X,Y,kernel,noise_var=0.1)
print(m.predict(X[:5,:]))
