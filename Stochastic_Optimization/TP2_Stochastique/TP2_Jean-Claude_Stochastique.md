![](https://deeptechbytes.com/wp-content/uploads/2021/02/Stochastic-optimisation.jpg)


>  # **<p align = center>Tp2</p>** 
> - # **Author: MITCHOZOUNOU Jean-CLaude**
>   - ## **Institute: UM6P**

<br />

---
# **<p align = "center">Newton avec région de confiance (Cas multidimensionnel)<p/>**
---

## 1- Gradient conjugué linéaire

>  1.1 **Gradient simple**

```sci
Grad_Mat.sci


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



```

>Output
![alt text](image.png)

---

>  1.2 **Gradient conjugué linéaire avec matrice**

```sci
GC_Mat.sci


function [dN,dNQ,niter,L_f] = GC_TR(c,Hv,x,eps,Delta,MaxIter,verbose)
    // résout le système linéaire Hv*dN+b=0 par la descente du gradient simple.
    // on présume que Hv est la matrice hessienne et c le gradient d'une fonction
    // à  minimiser. 
    // 
    //
    // Entrée:
    //    c(1,n), Hv(n,n) paramètres de la fonction quadratique à minimiser
    //        q(delt) = 0.5*delt'*Hv*delt + c*delt
    //        nabla q(delt) = delt'*Hv + c
    //        x  la variable dont depend Hv
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
      
      
      deltQ = Hv(x,delt);
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
      dQ = Hv(x,d);
      dQd = dQ*d;
        
      norm2nablaq = nablaq*nablaq';
      pasprecis = norm2nablaq>eps^2;



      sortie = %F
      while (~sortie & pasprecis) & (niter < MaxIter)
        niter = niter + 1;
        
        dQ = Hv(x,d);
        dQd = dQ*d;
        
        
        // Calcul de thetamax: Resoudre || delt + theta*d||<=Delta
        A = d'*d
        B = 2*(d'*delt)
        C = delt'*delt - Delta
        Discriminant = B^2 -4*A*C
        if Discriminant <0
            sortie = %T
        else
            racines = [-B-sqrt(Discriminant)/(2*A),-B+sqrt(Discriminant)/(2*A)] // Racine
            thetamax = max(racines)
            if thetamax<0
                thetamax = 1
            end
            theta=(-nablaq*d)/(dQd);
            if (theta<=0.0)|| (theta>thetamax)
                theta = thetamax
            end
            delt = delt + theta*d;
            deltQ = deltQ + theta*dQ;
            nablaq = nablaq + theta*dQ;
            
            bet = (dQ*nablaq')/(dQd)
            d = -nablaq' + bet*d;
           
            norm2nablaq = nablaq*nablaq'
            pasprecis = norm2nablaq>eps^2
            
        end

        
        if verbose>0,
       mprintf("%s %3d  %10.7f  %10.7e\n",pad,niter, (c*delt + 0.5*deltQ*delt), ...
              norm(nablaq))
        end
        L_f(niter) = c*delt + 0.5*deltQ*delt;
      end
      dN = delt;
      dNQ = deltQ;
endfunction

```
>Output
![alt text](image-1.png)

---

>  1.3 **Gradient conjugué linéaire sans matrice: La fonction**

```sci
GC_Hv.sci




function [dN,dNQ,niter,L_f] = GC_Hv(c,Hv,x,eps,MaxIter,verbose,delt0)
// résout le système linéaire Hv*dN+b=0 par la descente du gradient simple.
// on présume que Hv est la matrice hessienne et c le gradient d'une fonction
// à  minimiser. 
// 
//
// Entrée:
//    c(1,n), Hv(n,n) paramètres de la fonction quadratique à minimiser
//        q(delt) = 0.5*delt'*Hv*delt + c*delt
//        nabla q(delt) = delt'*Hv + c
//        x  la variable dont depend Hv
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
  
  
  deltQ = Hv(x,delt);
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
    
    dQ = Hv(x,d);
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
 

```
> Output
![alt text](image-2.png)

## Appel simultané de GC_Mat et GC_Hv
![alt text](image-3.png)
> norm(dNCGMat - dNCGHv)
![alt text](image-4.png)

---
---

## 2- Région de confiance 

> 2.1 **Adaptation de** *GC_Hv : GC_TR.sci*

```sci


function [dN,dNQ,niter,L_f] = GC_TR(c,Hv,x,eps,Delta,MaxIter,verbose)
    // résout le système linéaire Hv*dN+b=0 par la descente du gradient simple.
    // on présume que Hv est la matrice hessienne et c le gradient d'une fonction
    // à  minimiser. 
    // 
    //
    // Entrée:
    //    c(1,n), Hv(n,n) paramètres de la fonction quadratique à minimiser
    //        q(delt) = 0.5*delt'*Hv*delt + c*delt
    //        nabla q(delt) = delt'*Hv + c
    //        x  la variable dont depend Hv
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
      
      
      deltQ = Hv(x,delt);
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
      dQ = Hv(x,d);
      dQd = dQ*d;
        
      norm2nablaq = nablaq*nablaq';
      pasprecis = norm2nablaq>eps^2;



      sortie = %F
      while (~sortie & pasprecis) & (niter < MaxIter)
        niter = niter + 1;
        
        dQ = Hv(x,d);
        dQd = dQ*d;
        
        
        // Calcul de thetamax: Resoudre || delt + theta*d||<=Delta
        A = d'*d
        B = 2*(d'*delt)
        C = delt'*delt -Delta^2
        Discriminant = B^2 -4*A*C
        if Discriminant <0
            sortie = %T
        else
            racines = [-B-sqrt(Discriminant)/(2*A),-B+sqrt(Discriminant)/(2*A)] // Racine
            thetamax = max(racines)
            if thetamax<0
                //sortie = %T
                thetamax = 1
            end
            theta=(-nablaq*d)/(dQd);
            if (theta<=0.0)|| (theta>thetamax)
                theta = thetamax
            end
            delt = delt + theta*d;
            deltQ = deltQ + theta*dQ;
            nablaq = nablaq + theta*dQ;
   
            bet = (dQ*nablaq')/(dQd)
            d = -nablaq' + bet*d;
       
           
            norm2nablaq = nablaq*nablaq'
            pasprecis = norm2nablaq>eps^2
            
        end

          
        if verbose>0,
       mprintf("%s %3d  %10.7f  %10.7e\n",pad,niter, (c*delt + 0.5*deltQ*delt), ...
              norm(nablaq))
        end
        L_f(niter) = c*delt + 0.5*deltQ*delt;
      end
      dN = delt;
      dNQ = deltQ;
endfunction

```
> - Exécution des  trois instances de problèmes oú la taille de région de confiance est ajustéé pour valider que votre implantation
semble fournir des r ́esultats corrects
```sci
// Générateur de problèmes 

n = 15;

// valeurs propres extrêmes
lambda_1 = 1;
lambda_n = 20;
range = lambda_n-lambda_1;

lambda = lambda_1:range/(n-1):lambda_n
// Matrice diagonale avec les valeurs propres étalées entre lambda_1 et lambda_n
Lambda = diag(lambda);

// Générons une matrice de rotation "aléatoire".
M=rand(n,n);
[Q,R] = qr(M);

// H est une rotation de la matrice diagonale Lambda
H = Q*Lambda*Q';

// un vecteur b aléatoire
b = rand(1,n);
delt0=zeros(n,1);


function [Hv] = Hv(x,v)
  Hv = v'*H;
endfunction

verbose = 1;
MaxIter = 20;

x = zeros(1,n);
v = 1
exec ("GC_Hv.sci",0);

[dNCGHv,dNQ,ngc,L_fGC] = GC_Hv(b,Hv,x,1e-8,MaxIter,verbose,delt0);


exec ("GC_TR.sci",0);

Delta = (1.01)*norm(dNCGHv);

[dNTR,dNQ,ngc] = GC_TR(b,Hv,x,1e-8,Delta,MaxIter,verbose);

nor = norm(dNCGHv-dNTR) // devrait être exactement 0

disp(nor)

```
> 
![alt text](image-5.png)
![alt text](image-6.png)

```sci
Delta = (0.99)*norm(dNCGHv);

[dNTR,dNQ,ngc] = GC_TR(b,Hv,x,1e-8,Delta,MaxIter,verbose);

disp(norm(dNCGHv-dNTR)) // devrait être proche de 0

```
![alt text](image-7.png)

```sci
Delta = (0.5)*norm(dNCGHv);

[dNTR,dNQ,ngc] = GC_TR(b,Hv,x,1e-8,Delta,MaxIter,verbose);

disp(norm(dNCGHv-dNTR)) // devrait être loin de 0

```

![alt text](image-8.png)

> 2.2 **Test de l’algorithme**

```sci

```
