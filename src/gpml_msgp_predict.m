function [ym, ys2] = gpml_msgp_predict(hyp, x, y, xs, n_grid)

  xg = apxGrid('create',[x;xs],true,n_grid);
  
  dim = size(x, 2)
  %cov = {{@covSEiso},{@covSEiso}}; 
  %cov
  cov = cell(dim, 1);
  for i = 1:dim
   cov(i,1) = {{@covSEiso}};
  endfor
  
  covg = {@apxGrid,cov,xg};
  mean = {@meanZero}; lik = {@likGauss};

  opt.cg_maxit = 200; opt.cg_tol = 5e-3;
  infg = @(varargin) infGaussLik(varargin{:},opt);
  hyp
  [post,nlZ,dnlZ] = infGrid(hyp,{@meanZero},covg,{@likGauss},x,y,opt);
  [fm,fs2,ym,ys2] = post.predict(xs);
