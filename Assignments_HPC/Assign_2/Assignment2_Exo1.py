import numpy as np 
import time as tm
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
Nproc = COMM.Get_size()



# Functions 

def compute_points(param): 
    circle_points= 0
    # Total Random numbers generated= possible x 
    # values* possible y values 
    for i in range(param**2): 
        
        rand_x= np.random.uniform(-1, 1) 
        rand_y= np.random.uniform(-1, 1) 
        # Distance between (x, y) from the origin 
        origin_dist= rand_x**2 + rand_y**2
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1: 
            circle_points+= 1
    return circle_points




## Parallel version of Monte Carlo for Pi.

def Pi_MonteCarlo_parallel(N):
    # N: Total number of point
    t1 = tm.process_time()

    # Dispashing of N to the available processes.

    Np = N// Nproc
    Numbers = [Np + N%Nproc]
    for _ in range(1,Nproc):
        Numbers.append(Np) #Numbers is a list wich contains the numbers
                        # of point of each process
    Sum_square = sum([n**2 for n in Numbers])
    # Send of Np to each processes
    if RANK == 0:
        sendbuf = Numbers
    else:
        sendbuf = None

    recevbuf = COMM.scatter(sendbuf, root = 0)
    COMM.Barrier()

    # Recuperation of data
    Proc_Circle_points = COMM.gather(compute_points(recevbuf), root = 0)
    # Proc_Circle_points is a list of number of circle_point by processes
    COMM.Barrier()

    if RANK == 0:
        P = sum(Proc_Circle_points)
        t2 = tm.process_time()
        print(f"Pour Nbr_points= {N**2}, approximativement Pi={(4*P)/Sum_square} avec un temp de calcul Tcompute = {round(t2-t1,4)} s")
    
    #print(f"Process {RANK}: {recevbuf}") # Pour afficher la répartition des points par processes
       
    return 0



## Test


N = [10**k for k in range(1,4)]
pi = 0
for n in N:

    if Nproc ==1:
        t1 = tm.process_time()
        pi = (4*compute_points(n))/n
        t2 = tm.process_time()
        print(f"Pour N= {n**2}, approximativement Pi={pi} avec un temp de calcul Tcompute = {round(t2-t1,4)}s")
    else:
        Pi_MonteCarlo_parallel(n)
        
