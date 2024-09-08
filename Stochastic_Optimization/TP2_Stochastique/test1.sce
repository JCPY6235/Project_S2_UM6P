// reproduit les calculs du tableau de  f(x,y)=2x^2-2xy+y^2+2x-2y

H=[4 -2;-2 2];
b=[2 -2];
n=2;
delt0=[10;5];

xstar = -H\b';  // Solution optimale

verbose = 1;
MaxIter = 20;

exec ("Grad_Mat.sci",-1);


[dNG,dNQ,ngc,L_f] = Grad_Mat(b,H,1e-8,MaxIter,verbose,delt0);


