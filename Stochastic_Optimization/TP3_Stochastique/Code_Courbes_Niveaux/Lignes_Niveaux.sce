// Définissons la fonction f(x) et les paramètres
function f = f(x, y)
    f = 2*x^2 + 3*x*y + 2*y^2; // Exemple de fonction
endfunction


rho = linspace(10, 1000, 20); // Valeurs de rho


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



// Les lignes de niveaux
x1 = linspace(-5, 5, 100); // Plage de valeurs pour x1
x2 = linspace(-5, 5, 100); // Plage de valeurs pour x2
[X, Y] = ndgrid(x1, x2); // Grille 

for ho = 1:length(rho)
    
    Z = Phi(X,Y,ho);
    
    contour(x1, x2, Z, [1:7]);
    xlabel('x1');
    ylabel('x2');
    title('Lignes de niveaux de la function Varphi');
end

