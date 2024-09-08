function  [dN,dNQ,niter] = NewtonRegion_conf(c,Hv,x,eps,delta,MaxIter)
    //global nf ng nh;
    
    
    M=[];
    Delta = delta;
    iter = 0;
    pred = 0;
    ared = 0;
    lab = ["Iter", "f(x)","  " "|df(x)|", "Del", "Pred"," ", "Ared"]
    // Cout (fk), dérivées première (gk) et seconde (hk) au point de départ
    //disp(size(c))
    df = gradRosen(x);
    //img = f(x)
    img = f(x)
    
    termine = norm(df)<=eps  |  iter >= MaxIter
    disp(lab)
    while (~termine) | (iter == 0),
      l = [iter, img,  norm(df), Delta, pred, ared];
      M($+1,:)=l;
     // disp(size(trans(df)))
      
      deff("[qq]=q(d)"," qq=img + df*d + 0.5*Hv(x,d)*d");
      // le code de l'algorithme vient ici
      
       dr = GC_TRosen(c,Hv,x,eps,Delta,MaxIter);
       //disp(size(dr))
      //disp(size(img))
      
      img1 = f(x+dr')
      ared = img  - img1;
      //z = 0*dr;
      //disp(img)
      pred = q(zeros(2,1)) - q(dr);
      r = ared/pred;
      //disp(r)
      if r < 0.25,
          Delta = Delta/2;
      else,
          x = x + dr';
      end
      
      if r > 0.75,
          Delta = 2*Delta;
      end
      
      iter=iter+1;  
     
      //disp(iter, x, abs(gk), del, pred , ared)
      df = gradRosen(x);
      img = f(x)
      img1 = f(x+dr')
     
    l = [iter, img,  norm(df), Delta, pred, ared];
    M($+1,:)=l;
      termine = norm(df)<=eps  |  iter >= MaxIter;
    end;  //eps1 = 0.2;
  
      
    
    dN = x;
    niter = iter;
    dNQ = df;
    disp(M)
    disp(x)
  endfunction
  
  
