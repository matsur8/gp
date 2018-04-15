function [hyp] = gpml_basic_make_model(hyp, x, y)

  meanfunc = [];                    
  covfunc = @covSEiso; % Squared Exponental covariance function
  likfunc = @likGauss; % Gaussian likelihood
  hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);


