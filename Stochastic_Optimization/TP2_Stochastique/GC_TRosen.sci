
function [dN] = GC_TRosen(c,Hv,x,eps,Delta,MaxIter)
    
      
    niter = 0; 
    sortie = 0;
    thetamax = 0;
    
    L_f = [];
    [bidon,dim] = size(c)
    
    delt = delt0;
    
    
    deltQ = Hv(x,delt);
    nablaq = c + deltQ;
    
    d = -nablaq'
    dQ = Hv(x,d);
    dQd = dQ*d;
      
    norm2nablaq = nablaq*nablaq';
    pasprecis = norm2nablaq>eps^2;

    A = d'*d
    B = 2*(d'*delt)
    C = delt'*delt -Delta^2
    Discriminant = B^2 -4*A*C
    
    if Discriminant <0
        sortie = 1
    else
        racines = [-B-sqrt(Discriminant)/(2*A),-B+sqrt(Discriminant)/(2*A)] // Racine
        thetamax = max(racines)
    end
    if thetamax<=0
        //sortie = 1
        thetamax = 1
    end

    theta=(-nablaq*d)/(dQd);
    if (theta<=0.0)|| (theta>thetamax)
        theta = thetamax
    end


    while (sortie == 0 )& pasprecis & (niter < MaxIter)
      niter = niter + 1;
      
      //if dQd<0.0
      //sortie = 1;
     // end
      
      
      delt = delt + theta*d;
      deltQ = deltQ + theta*dQ;
      nablaq = nablaq + theta*dQ;

      bet = (dQ*nablaq')/(dQd)
      d = -nablaq' + bet*d;
      dQ = Hv(x,d);
      dQd = dQ*d;
    
      norm2nablaq = nablaq*nablaq'
      pasprecis = norm2nablaq>eps^2
      
      // Calcul de thetamax: Resoudre || delt + theta*d||<=Delta
      A = d'*d
      B = 2*(d'*delt)
      C = delt'*delt -Delta^2
      Discriminant = B^2 -4*A*C
      
      if Discriminant <0
          sortie = 1
      else
          racines = [-B-sqrt(Discriminant)/(2*A),-B+sqrt(Discriminant)/(2*A)] // Racine
          thetamax = max(racines)
      end
      if thetamax<0
          thetamax = 1
         // sortie = 1
          
      end

      theta=(-nablaq*d)/(dQd);
      if (theta<=0.0)| (theta>thetamax)
          theta = thetamax
      end


      
      
    //L_f(niter) = c*delt + 0.5*deltQ*delt;
    end
    dN = delt;
    //dNQ = deltQ;
endfunction

