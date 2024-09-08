
function [dN,dNQ,niter,L_f] = Grad_Mat(c,Q,eps,MaxIter,verbose,delt0)
// résout le système linéaire Q*dN+b=0 par la descente du gradient simple.
// on présume que Q est la matrice hessienne et c le gradient d'une fonction
// à  minimiser. 
// 
//
// Entrée:
//    c(1,n), Q(n,n) paramètres de la fonction quadratique à minimiser
//        q(delt) = 0.5*delt'*Q*delt + c*delt
//        nabla q(delt) = delt'*Q + c
//    eps tolérance d'arrêt
//    MaxIter  nombre maximum d'itérations
//    verbose  imprime les itérations (>0) ou non (<=0)
//    delt0(n,1)  point de départ des itérations
//
// Sortie:
//    dN: la solution  :dN*Q+b=0
//    niter: nombre total d'itérations
//    L_f: liste des valeurs de fonctions au fil des itérations
//   
  
  niter = 0; 
  L_f = [];
  [bidon,dim] = size(c)
  
  delt = delt0;
  deltQ = delt'*Q;
  nablaq = c + deltQ;
  
  pad = '';
  if verbose >0,
    s='   ';
    for i=1:verbose,
      pad = pad + s
    end
    mprintf("%s iter    q(delt)  ||nabla q(delt)||\n",pad)
    mprintf("%s %3d  %10.7f  %10.7e\n",pad,niter, (c*delt + 0.5*deltQ*delt), ...
	    norm(nablaq))
  end
  
  norm2nablaq = nablaq*nablaq';
  pasprecis = norm2nablaq>eps^2;
  
  while pasprecis & (niter < MaxIter)
    niter = niter + 1;
    p = -nablaq';
    pQ = p'*Q;
    pQp = pQ*p
    
    theta=norm2nablaq/(pQp);
    if theta<0.0
      warning("Q not positive ")
    end
    
    delt = delt + theta*p;
    deltQ = deltQ + theta*pQ;
    nablaq = nablaq + theta*pQ;
    
    norm2nablaq = nablaq*nablaq'
    pasprecis = norm2nablaq>eps^2;
    if verbose>0,
      mprintf("%s %3d  %10.7f  %10.7e\n",pad,niter, (c*delt + 0.5*deltQ*delt), ...
	      norm(nablaq))
    end
    L_f(niter) = c*delt + 0.5*deltQ*delt;
  end
  dN = delt;
  dNQ = deltQ;
endfunction
 
 
 // reproduit les calculs du tableau de  f(x,y)=2x^2-2xy+y^2+2x-2y

H=[4 -2;-2 2];
b=[2 -2];
n=2;
delt0=[10;5];

xstar = -H\b';  // Solution optimale

verbose = 1;
MaxIter = 20;




[dNG,dNQ,ngc,L_f] = Grad_Mat(b,H,1e-8,MaxIter,verbose,delt0);



