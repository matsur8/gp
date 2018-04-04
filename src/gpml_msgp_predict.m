function [ym, ys2] = gpml_msgp_predict(x, y, hyp, xs)

  xg = apxGrid('create',[x;xs],true,[50,70]);

  cov = {{@covSEiso},{@covSEiso}}; covg = {@apxGrid,cov,xg};
  mean = {@meanZero}; lik = {@likGauss};

  opt.cg_maxit = 200; opt.cg_tol = 5e-3;
  infg = @(varargin) infGaussLik(varargin{:},opt);

  [post,nlZ,dnlZ] = infGrid(hyp,{@meanZero},covg,{@likGauss},x,y,opt);
  [fm,fs2,ym,ys2] = post.predict(xs);
