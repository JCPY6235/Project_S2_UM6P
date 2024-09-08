//Penality function
function [p]= P(x)
    v = ((x(1) + 0.5)*( x(1) > -0.5))**2 + ((x(2) + 0.5)*( x(2) > -0.5))**2
    p = v,
endfunction



//Main function and gradient
function [df, obj]=f_grad(x, sigma)
    f_x= 4*x(1) + 3*x(2) + sigma*(x(1) + 0.5)*( x(1) > -0.5),
    f_y=  4*x(2) + 3*x(1) + sigma*(x(2) + 0.5)*( x(2) > -0.5),
    
    f = 2*x(1)**2 + 2*x(2)**2 + 3*x(1)*x(2) + 0.5*sigma*P(x)
    
    
    df = [f_x; f_y]
    obj = f
endfunction

function [dff] = Hv(x,v, sigma)
    n = size(v)(1)
    G = zeros(n,n)
// Hessien matrix 
    G(1,2) = 3
    G(2,1) = 3
    G(1,1) = 4 + sigma*(x(1) > -0.5)
    G(2,2) = 4 + sigma*(x(2) > -0.5)
    
    dff = v'*G;
  endfunction
  


verbose = 1
// starting point
xo = [-0.3; 0.5]

//Parameters
c = 2
rho = 10000
eps2 = 10^-5
eps1 = 10^-6



//Calling of the function
exec ("Algo_PénalitéExtérieur.sce",0);

[xop, fopt, dN, it] = Algo_PénalitéExtérieur(xo, eps1, eps2, c, rho, P, f_grad, 20)

mprintf("X optimal %10.3e  %10.3e\n\n",xop(1),xop(2))
