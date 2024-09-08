
function [dN] = GC_TR(c,Hv,x,eps,d,MaxIter, sigma)
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
  sortie = 0;
  niter = 0; 
 // L_f = [];
  [n,dim] = size(c')
  
  delt = zeros(n,1);
  deltQ = Hv(x,delt, sigma);
  nablaq = c + deltQ;
  
  
  norm2nablaq = nablaq*nablaq';
  pasprecis = norm2nablaq>eps^2;
  p = -nablaq';
  pQ = Hv(x,p, sigma);
  pQp = pQ*p
 
  pdelt = p'*delt
  
  Del = pdelt^2 - (p'*p)*(delt'*delt - d^2);
  if Del < 0;
    sortie = 1;
  else,
    Theta_max = (-pdelt + sqrt(Del))/(p'*p);
  end
  
  if Theta_max <= 0 then
    sortie = 1;
  end
    
  Theta = - nablaq*p/pQp
  if (Theta > Theta_max) | (Theta < 0) then
      Theta = Theta_max
  end
  while pasprecis & (niter < MaxIter) & (sortie == 0)
    niter = niter + 1; 
    
    
    theta=pQp;
    if theta<0.0
      //warning("Bad direction ")
      sortie = 1;
    end
    

    delt = delt + Theta*p;
    
    deltQ = deltQ + Theta*pQ;
    nablaq = nablaq + Theta*pQ;
    bet = Hv(x,nablaq', sigma)*p/pQp;
    
    p = -nablaq' + bet*p;
    pQ = Hv(x,p,sigma);
    pQp = pQ*p;
    norm2nablaq = nablaq*nablaq';
    pasprecis = norm2nablaq>eps^2;
    pdelt = p'*delt
    
    Del = pdelt^2 - (p'*p)*(delt'*delt - d^2);
    if Del < 0;
        sortie = 1;
    else,
        Theta_max = (-pdelt + sqrt(Del))/(p'*p);
    end
  
    if Theta_max <= 0 then
        sortie = 1
    end
    
    Theta = - nablaq*p/pQp
    if (Theta > Theta_max) | (Theta < 0) then
        Theta = Theta_max
    end
  
  end
  dN = delt;
  
endfunction

