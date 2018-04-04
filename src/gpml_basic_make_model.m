function [hyp] = gpml_basic_make_model(hyp, x, y)

  addpath(genpath("gpml"))

  meanfunc = [];                    % empty: don't use a mean function
  covfunc = @covSEiso;              % Squared Exponental covariance function
  likfunc = @likGauss;              % Gaussian likelihood
  hyp = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);


