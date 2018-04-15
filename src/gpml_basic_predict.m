function [ym, ys2] = gpml_basic_predict(hyp, x, y, xs)

  meanfunc = [];                    
  covfunc = @covSEiso; % Squared Exponental covariance function
  likfunc = @likGauss; % Gaussian likelihood

  [ym, ys2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs); %predict

