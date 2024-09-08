// Générateur de problèmes 

n = 15;

// valeurs propres extrêmes
lambda_1 = 1;
lambda_n = 20;
range = lambda_n-lambda_1;

lambda = lambda_1:range/(n/3-1):lambda_n
// Matrice diagonale avec les valeurs propres étalées entre lambda_1 et
// lambda_n répétés trois fois

Lambda = diag([lambda lambda lambda]);

// Générons une matrice de rotation "aléatoire".
M=rand(n,n);
[Q,R] = qr(M);

// H est une rotation de la matrice diagonale Lambda
H = Q*Lambda*Q';

// un vecteur b aléatoire
b = rand(1,n);
delt0=zeros(n,1);


verbose = 1;
MaxIter = 20;

exec("GC_Mat.sci",0);

[dNCGMat,dNQ,ngc,L_fGC] = GC_Mat(b,H,1e-8,MaxIter,verbose,delt0);



function [Hv] = Hv(x,v)
  Hv = v'*H;
endfunction

exec ("GC_Hv.sci",0);

x = zeros(1,n); 
[dNCGHv,dNQ,ngc,L_fGC] = GC_Hv(b,Hv,x,1e-8,MaxIter,verbose,delt0);

disp(norm(dNCGMat - dNCGHv)) // devrait être exactement 0

