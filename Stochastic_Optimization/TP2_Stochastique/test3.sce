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
exec ("GC_Hv.sci",0);

[dNCGHv,dNQ,ngc,L_fGC] = GC_Hv(b,Hv,x,1e-8,MaxIter,verbose,delt0);


exec ("GC_TR.sci",0);


Delta = (1.01)*norm(dNCGHv);

[dNTR,dNQ,ngc,L_fTR] = GC_TR1(b,Hv,x,1e-8,Delta,MaxIter,verbose);

nor = norm(dNCGHv-dNTR) // devrait être exactement 0

disp(nor)


Delta = (0.99)*norm(dNCGHv);

[dNTR,dNQ,ngc] = GC_TR1(b,Hv,x,1e-8,Delta,MaxIter,verbose);

disp(norm(dNCGHv-dNTR)) // devrait être proche de 0


Delta = (0.5)*norm(dNCGHv);

[dNTR,dNQ,ngc] = GC_TR1(b,Hv,x,1e-8,Delta,MaxIter,verbose);

disp(norm(dNCGHv-dNTR)) // devrait être loin de 0
