// La fonction f(x, y)

function f = f(x, y)
    f = 2*x^2 + 3*x*y + 2*y^2; // Exemple de fonction
endfunction

// La fonction phi sur X

function fi = Phi(x,y, ho)
    fval = f(x,y);
    fi = fval;
endfunction



// Calculer les composantes du gradient de phi(x, y)

function [fx,fy] = Grad_Phi(x,y, ho)
     fx = 4*x + 3*y;
     fy = 4*y + 3*x ;
endfunction


rho = linspace(10, 1000, 20); // Valeurs de rho

// Plage de valeurs pour x et y dans le domaine x
x = linspace(-5, 0, 50);
y = linspace(-5, 0, 50);

// Créer une grille de valeurs pour x et y
[X, Y] = ndgrid(x, y);


for ho = 1:length(rho)
    // Calculer les composantes du gradient pour chaque point de la grille
    [grad_x, grad_y] = Grad_Phi(X, Y,ho);

    // Tracer les courbes de niveaux du gradient
    contour(x, y, sqrt(grad_x.^2 + grad_y.^2),[1:5]);
    xlabel('x');
    ylabel('y');
    title('Courbes de niveaux du gradient de Phi sur X');
end
