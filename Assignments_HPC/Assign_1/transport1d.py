# Ce programme résoud une équation de transport 1D par la méthode 
# des différences finies
import numpy as np
import matplotlib.pyplot as plt
from source import f

# Longueur du canal
L = 1000
# Nombre de noeuds du maillage
N = 200
# Le pas du maillage
dx = L/(N-1)
# Vitesse du transport
a = 1

# Maillage et condition initiale
x = np.linspace(0,L,N)
u = np.zeros(N)
for i in range(N):
    u[i] = f(x[i])
# Tracé de la condition initiale
plt.plot(x,u,'-b')
plt.grid()

# Temps final des simulations
Tfinal = 1000
# Initialisation du temps
temps = 0
# Nombre CFL tel que (0 < CFL <=1)
CFL = 0.9
# Calcul du pas du temps pour assurer la stabilité
dt = CFL*dx/abs(a)

lamda = a*dt/dx
unew=np.zeros(N)
# Boucle principale en temps
while (temps < Tfinal):
    for i in range(1,N-1):
        unew[i] = u[i]-lamda*(u[i]-u[i-1])        # schéma décentré amont (si a > 0)
        #unew[i] = u[i]-lamda*(u[i+1]-u[i])        # schéma décentré aval (si a < 0)
        #unew[i] = u[i]-(lamda/2)*(u[i+1]-u[i-1])  # schéma centré (instable) 
        #unew[i] = (u[i-1]+u[i+1])/2  \
        #                -(lamda/2)*(u[i+1]-u[i-1]) # Law-Friedrichs
        #unew[i] = u[i]-(lamda/2)*(u[i+1]-u[i-1]) \
        #              +(lamda**2/2)*(u[i-1]-2*u[i]+u[i+1]) # Lax-Wendroff
    # Conditions aux limites de (périodique à l'entrée et Neumann à la sortie)
    unew[0] = unew[N-1]
    unew[N-1] = unew[N-2]
    # Incrémentation du temps et mise à jour du tableau u
    temps += dt
    u = unew.copy()
    # Courbes de u au cours du temps    
    plt.plot(x,u,'-r')
        #plt.axis([0,1000,0,11])
    plt.grid()
    plt.pause(0.01)
        



    