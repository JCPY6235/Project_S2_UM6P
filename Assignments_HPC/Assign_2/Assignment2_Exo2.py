from mpi4py import MPI
import numpy as np
import time
import warnings

Maxiter = 100
np.random.seed(0)
X = np.random.ranf(1000)
noise = [np.random.normal() for i in range(np.size(X)) ]

f = lambda X: 2*X
# Labels
Y = f(X) + noise

#plt.scatter(X,Y, s = 0.2)
weight = np.ones(2)

def cost(X,Y, weight):
    n = np.size(X)
    cost_ = np.dot(X,weight[0]) + weight[1]
    err = cost_ - Y
    return (1/n)*np.dot(err,err)
# Calcul du gradient 

def Compute_Gadient(weight, data = X, labels = Y):
    n = np.size(X)
    predict = weight[0]*data + weight[1]
    err = predict-labels
    grad1 = 2*np.dot(data,err)
    grad2 = 2*np.sum(err)
    return np.array((grad1,grad2))




#### --------------------Gradient descent stochastique ----------------####

def Parallel_Stochastic_Gradient(Maxiter,modelparam,alph,data = X,labels = Y):
    COMM = MPI . COMM_WORLD
    Nproc = COMM . Get_size()
    RANK = COMM . Get_rank()
    iter = 0
    
    N = np.size(data)
    r = N%Nproc
    ### Spliting of datasets distribution among processes

    if RANK == 0:
        n_proc = N//Nproc+r
        start = RANK*n_proc
        end = n_proc
    else:
        n_proc = N//Nproc
        start = n_proc*RANK +r
        end = n_proc*(RANK+1)+r

    Weight = modelparam
    t1 = time.process_time()
    #warnings.filterwarnings("ignore", category=DeprecationWarning)
    for j in range(Maxiter):
    
        # Calcul du gradient locallelement et récupération

        Loc_gradients =Compute_Gadient(Weight,X[start:end],Y[start:end])
        #print(f"{RANK}:Loc_gradients = {Loc_gradients}")
        # Gradient Global
        gradient=COMM.allreduce(Loc_gradients, op=MPI.SUM)/N

        COMM.Barrier()
        
        

        Weight =  Weight-alph*gradient
        #print(cost(X,Y,Weight))
        iter+=1
    t2 = time.process_time()
    if RANK ==0:
        print(f"La droite de regression a pour paramètre:(a,b) = {Weight}")
    
    


### Test

Parallel_Stochastic_Gradient(Maxiter,weight,0.3,X,Y)
