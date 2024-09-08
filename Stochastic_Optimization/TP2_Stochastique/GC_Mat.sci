

function [dN,dNQ,niter,L_f] = GC_Mat(c,Q,eps,MaxIter,verbose,delt0)
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
  d = -nablaq'
  bet = 0 
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
    d = -nablaq' + bet*d;
    
    dQ = d'*Q;
    dQd = dQ*d;
    
    theta=(-nablaq*d)/(dQd);
    
    if theta<0.0
      warning("Q not positive ")
    end
    
    delt = delt + theta*d;
    deltQ = deltQ + theta*dQ;
    nablaq = nablaq + theta*dQ;
    
    bet = (dQ*nablaq')/(dQd)
    
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
 
