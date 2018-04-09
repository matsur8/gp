function [hyp] = gpml_msgp_make_model(hyp, x, y, n_grid)
  %addpath(genpath("gpml"))

  xg = apxGrid('create', x, true, n_grid);

  dim = size(x, 2);
  cov = cell(dim, 1);
  for i = 1:dim
    cov(i,1) = {{@covSEiso}};
  endfor

  covg = {@apxGrid,cov,xg};
  mean = {@meanZero}; 
  lik = {@likGauss};
  %hyp.cov = zeros(2*dim,1); hyp.mean = []; hyp.lik = log(0.1);
  hyp.proj = [1; 1];
  opt.cg_maxit = 200; 
  opt.cg_tol = 5e-3;
  opt.pred_var = 500;
  opt.ndcovs = 20;
  opt.proj = "norm"
  infg = @(varargin) infGaussLik(varargin{:},opt);
  hyp = minimize(hyp,@gp,-10,infg,[],covg,[],x,y);
 

