function [ym, ys2] = gpml_msgp_predict(hyp, x, y, xs, n_grid)

  xg = apxGrid('create',[x;xs],true,n_grid);
  %randn("state", [1;3;4;2;1]);
  dim = size(x, 2)
  cov = cell(dim, 1);
  for i = 1:dim
   cov(i,1) = {{@covSEiso}};
  endfor
  
  covg = {@apxGrid,cov,xg};
  mean = {@meanZero}; 
  lik = {@likGauss};

  opt.cg_maxit = 200;
  opt.cg_tol = 5e-3;
  opt.pred_var = 500
  opt.ndcovs = 20
  %infg = @(varargin) infGaussLik(varargin{:},opt);
  [post,nlZ,dnlZ] = infGrid(hyp,{@meanZero},covg,{@likGauss},x,y,opt);
  [fm,fs2,ym,ys2] = post.predict(xs);
