import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import process_time_ns


COMM = MPI.COMM_WORLD
nproc = COMM.Get_size()
RANK = COMM.Get_rank()

# Longueur du canal
L = 1000
# Temps Final
nt = 100
# Nombre de noeuds du maillage
nx = 200
# Le pas du maillage
dx = L/(nx-1)

# Vitesse du transport
c = 1
# CFL 
CFL = 1
dt = CFL*dx/c


def f(x):
    if (300<=x<=400):
        return 10
    return 0

# Condition initiale
x = np.linspace(0,L,nx)
u0 = np.zeros(nx)
for i in range(nx):
    u0[i] = f(x[i])
# Tracé de la condition initiale
#plt.plot(x,u0,'-b')
#plt.grid()
#plt.show()





def solve_1d_linearconv(u, un, nt, nx, dt, dx, c):
    for n in range(nt):  
        for i in range(nx): un[i] = u[i]
        for i in range(1, nx): 
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
    return u

# Resolution parallèle

def solve_1d_linear_parallel(u, un, nt, nx, dt, dx, c):
    v = 0
    nloc = nx// nproc
    nlocP0 = nloc + (nx % nproc) # Au cas oú la division n'est pas exact , on étend le nombre
                # de noeuds du process 0 en y ajoutant le reste de la division
    
    u1 = np.zeros(nlocP0+1)
    u2 = np.zeros(nloc+1)
    U = np.empty(0) # Pour collecter les sous sections de la solution sur
        
    
    #COMM.Barrier()

    # Initialisation des données sur chaque process


    if RANK == 0:
        for i in range(nlocP0):
            u1[i] = u[i]
        u1[nlocP0] = u1[nlocP0-1]
        COMM.send(u1[nlocP0], dest=RANK+1)
        #v = u1[0:nlocP0]
        

    for j in range(1,nproc-1):  
        if RANK == j:
            u2[0] = COMM.recv(source = RANK-1)
            for i in range(nlocP0+(RANK-1)*nloc,nlocP0+RANK*nloc):
                u2[nlocP0+RANK*nloc-i] = u[i]
            u2[nloc] = u2[nloc-1]
            COMM.send(u2[nloc], dest=RANK+1)
            #v = u2[0:nloc]
            
    if RANK == nproc-1:
        u2[0] = COMM.recv(source = RANK-1)
        for i in range(nlocP0+(RANK-1)*nloc,nlocP0+RANK*nloc):
            u2[nlocP0+RANK*nloc-i] = u[i]
        #v = u2[0:nloc]
        


    # Calculs des valeurs u_n et échange des valeurs de bords

    for n in range(nt):
        if RANK == 0:
            for i in range(nlocP0): un[i] = u1[i]
            for i in range(1, nlocP0): 
                u1[i] = un[i] - c * dt / dx * (un[i] - un[i-1])

            u1[nlocP0] = u1[nlocP0-1]
            COMM.send(u1[nlocP0], dest=RANK+1)
            v = u1[0:nlocP0]
            z = len(v)
            
            
        
        for j in range(1,nproc-1):  
            if RANK == j:
                for i in range(nloc): un[i] = u2[i]
                for i in range(1, nloc): 
                    u2[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
                u2[0] = COMM.recv(source = RANK-1)
                u2[nloc] = u2[nloc-1]
                COMM.send(u2[nloc], dest=RANK+1)
                v = u2[0:nloc]
                z = len(v)



        if RANK == nproc-1:
            for i in range(nloc): un[i] = u2[i]
            for i in range(1, nloc): 
                u2[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            u2[0] = COMM.recv(source = RANK-1)
            v = u2[0:nloc]
            z = len(v)
            
        #COMM.Barrier()


# Racollement des vecteurs : On recolle tout via le process 0
    if RANK == 0:
        U = np.concatenate((U,v))
        for p in range(1,nproc):
            U = np.concatenate((U,COMM.recv(source = p, tag = 1)))
        plt.plot(x,U)
        plt.grid()
        plt.show()
        #print("La solution s'écrit: U = ",U)
    
    for k in range(1,nproc):
        if RANK == k:
            COMM.send(v,dest = 0, tag = 1)

    return z #U




# Découpage des tâches et attribution aux processes.
    
if nproc ==1: # Le cas oú il y un seul processeur
    nloc = nx
    un = np.zeros(nx)
    u = solve_1d_linearconv(u0,un,nt,nx,dt,dx,c)
    plt.plot(x,u0,'-r')
    plt.grid()
    plt.show()

else: # Au moins deux processeurs
    un = np.zeros(nx)
    u = np.array(solve_1d_linear_parallel(u0,un,nt,nx,dt,dx,c))




print(f"Nombre de noeuds process{RANK} = {u}")
