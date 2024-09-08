


// Générateur de problèmes 

n =2;

b = rand(1,n);
delt0=ones(n,1);


function [val] = f(x)
  val = 100*(x(2)-x(1)^2)^2 + (1-x(1))^2
endfunction

//
function [Hvr] = RosHv(x,v)
  Hvr = v'*[-400*(x(2)-x(1)^2)+800*x(1)^2+2 -400*x(1); -400*x(1) 200];
endfunction
// Gradient
function [grad] = gradRosen(x)
  grad = [-400*x(1)*(x(2)-x(1)^2) 200*(x(2)-x(1)^2)]
endfunction


verbose = 1;
MaxIter = 20;

xo = [0 1];

Delta = 0.5;
exec ("GC_TRosen.sci",0);
exec ("Newtom_MoRegionConf.sci",0);

[dN,dNQ,niter] = NewtonRegion_conf(b,RosHv,xo,1e-8,Delta,MaxIter);

