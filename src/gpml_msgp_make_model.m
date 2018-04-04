function [hyp] = gpml_msgp_make_model(x,y,grid)
  %addpath(genpath("gpml"))
  xg = apxGrid('create',x,true,grid);

  cov = {{@covSEiso},{@covSEiso}}; covg = {@apxGrid,cov,xg};
  mean = {@meanZero}; lik = {@likGauss};
  hyp.cov = zeros(4,1); hyp.mean = []; hyp.lik = log(0.1);

  opt.cg_maxit = 200; opt.cg_tol = 5e-3;
  infg = @(varargin) infGaussLik(varargin{:},opt);
  hyp = minimize(hyp,@gp,-10,infg,[],covg,[],x,y);

