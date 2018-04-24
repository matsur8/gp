import numpy as np

lb = -0.01
ub = 1.01
delta = 0.0002

def make_rbf_feature_1d(x, ls, v, delta, xgrid):
   return np.sqrt(delta * v) * (2.0 / (np.pi * ls**2))**0.25 * np.exp(-(xgrid - x)**2.0/ls**2.0)

def generate_data_1d(lengthscale, variance, noise_variance, n=None, X=None, random_state=None):
   if random_state is None:
      rs = np.random
   else:
      rs = random_state

   if X is None:
      X = rs.random_sample((n, 1))
   else:
      n, dim = X.shape

   xgrid = np.arange(lb, ub, delta)
   n_grid = xgrid.shape[0]
   w = rs.normal(size=n_grid)
   f = np.array([make_rbf_feature_1d(x[0], lengthscale, variance, delta, xgrid).dot(w) for x in X])
   y = f + np.sqrt(noise_variance) * rs.normal(size=n) 
   return X, y, f

if __name__ == "__main__":
   import matplotlib.pyplot as plt
   import GPy
   
   lengthscale = 0.0005
   variance = 1.2
   noise_variance = 0.8

   rs = np.random.RandomState(322)

   n = 60
   dim = 1
   X = rs.random_sample((n, dim)) * lengthscale * 10
   
   if n < 100:
      exact = True
      k = GPy.kern.RBF(input_dim=dim, lengthscale=lengthscale, variance=variance)
      cov = k.K(X,X) 
      f = rs.multivariate_normal(np.zeros(n), cov)
      y = f + np.sqrt(noise_variance) * rs.normal(size=n)
   
   X, y2, f2 =  generate_data_1d(lengthscale, variance, noise_variance, X=X, random_state=rs)

   if True:
      #plt.plot(X[:100,0], f[:100,0], "o")
      #plt.show()
      plt.plot(X[:,0], f2, "o")
      #plt.xlim((0, 5 * lengthscale))
      plt.show()

      #plt.plot(X[:100,0], y[:100,0], "o")
      #plt.show()
      plt.plot(X[:,0], y2, "o")
      #plt.xlim((0, 100 * lengthscale))
      plt.show()
      print(f2.var())
      print((f2**2).mean())


   if n < 100:
      print(f.var())
      print((f**2).mean())

      xgrid = np.arange(lb, ub, delta)
      cov2 = np.zeros((n, n))
      for i in range(n):
         fi = make_rbf_feature_1d(X[i,0], lengthscale, variance, delta, xgrid)
         for j in range(i, n):
            fj = make_rbf_feature_1d(X[j,0], lengthscale, variance, delta, xgrid)
            cov2[i,j] = fi.dot(fj)
            cov2[j,i] = cov2[i, j]


      print(np.max(np.abs(cov - cov2)))
 
