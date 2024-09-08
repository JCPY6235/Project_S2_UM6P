
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
            if (theta<=0.0)| (theta>thetamax)
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

