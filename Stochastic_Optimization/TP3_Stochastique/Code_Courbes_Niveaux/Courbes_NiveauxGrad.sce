// La fonction f(x, y)

function f = f(x, y)
    f = 2*x^2 + 3*x*y + 2*y^2; // Exemple de fonction
endfunction

// La fonction phi

function fi = Phi(x,y, ho)
    if (x<=-0.5) & (y<=-0.5) then
        fval = f(x,y);
        fi = fval;
    else
        fval = f(x,y);
        fi = fval + 0.5*ho*[(x+0.5)^2 + (y+0.5)^2];
    end
endfunction



// Calculer les composantes du gradient de phi(x, y)

function [fx,fy] = Grad_Phi(x,y, ho)
    if (x<=-0.5) & (y<=-0.5) then
        fx = 4*x + 3*y;
        fy = 4*y + 3*x;  
    else
        fx = 4*x + 3*y+ ho*(x+0.5);
        fy = 4*y + 3*x + ho*(y+0.5);
    end
endfunction


rho = linspace(10, 1000, 20); // Valeurs de rho

// Plage de valeurs pour x et y
x = linspace(-5, 5, 100);
y = linspace(-5, 5, 100);

// Créer une grille de valeurs pour x et y
[X, Y] = ndgrid(x, y);


for ho = 1:length(rho)
    // Calculer les composantes du gradient pour chaque point de la grille
    [grad_x, grad_y] = Grad_Phi(X, Y,ho);

    // Tracer les courbes de niveaux du gradient
    contour(x, y, sqrt(grad_x.^2 + grad_y.^2),[1:7]);
    xlabel('x');
    ylabel('y');
    title('Courbes de niveaux du gradient de Phi');
end
