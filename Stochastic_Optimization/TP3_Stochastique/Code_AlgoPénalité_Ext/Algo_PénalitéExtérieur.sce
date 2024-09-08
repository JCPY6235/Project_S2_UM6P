
function [xop, fopt, dN, it] = Algo_PénalitéExtérieur(xo, eps1, eps2, c, rho, P, f_grad, maxIter)
    xk = xo,
    rhok = rho
    k = 0
    M = 10^9
    cost = 0
    pad = '';
    
    if verbose >0,
            s='   ';
            for i=1:verbose,
              pad = pad + s
            end
        mprintf("%s iter    rhok      xk.\n",pad)
        mprintf("%s %3d  %10.1f  %10.3e  %10.3e\n",pad,k,rhok, xk(1),xk(2))
     end
        
    
    while rhok < M & abs(P(xk)) > eps1 & k < maxIter,
         
        j = 0
        xj = xk
        [Df, cost] = f_grad(xj, rhok)
        
        while norm(Df) > eps2,
            [Df, cost] = f_grad(xj, rhok),
            //dj = -Df // Direction standard
            dj = GC_TR(Df',Hv,xj, 1e-8, 0.8, maxIter, rhok) // Direction Newton Région de confiance
            theta = 1
            [f1, f2] =  f_grad(xj + theta*dj, rhok)
            to = 0.3
            
            while (f2 - cost) > theta*to*Df'*dj,
                theta = theta/2,
                
                [f1, f2] =  f_grad(xj + theta*dj, rhok)
            end
            
            xj = xj + theta*dj,
            j = j + 1
        end
        
        xk = xj,
        rhok = c*rhok,
        k = k + 1,
        
        if verbose >0,
            mprintf("%s %3d  %10.1f  %10.3e  %10.3e\n\n",pad,k,rhok,xk(1),xk(2))
        end
    
        
    end
    
    xop = xk
    fopt = cost
    it = k
    dN = f1
endfunction

 
