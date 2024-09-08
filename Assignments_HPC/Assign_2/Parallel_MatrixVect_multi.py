import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from mpi4py import MPI

''' This program compute parallel csc matrix vector multiplication using mpi '''

COMM = MPI.COMM_WORLD
Nproc = COMM.Get_size()
RANK = COMM.Get_rank()

def matrixVectorMult(A, b):
    row, col = A.shape
    x = np.zeros(row)
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]
    return x




def Parallel_Matrix_Vector_Mult(A,b):
    N = len(A) # Nombre de ligne de la matrice
    l = MPI.Wtime()
    n_proc = (N//Nproc) # Division suivant le nombre de processes
    r = N%Nproc # reste

    if RANK == 0:
        # La découpe de la matrix A.
        Mat =[A[0:n_proc+r,:]]
        for n in range(1,Nproc):
            Mat.append(A[(n*n_proc)+r:(n+1)*n_proc+r,:])
    else :
        Mat = None
        b = None



    LocalMatrix = COMM.scatter(Mat, root=0) # Local Matrix C
    b = COMM.bcast(b, root = 0)
    LocalVect = LocalMatrix.dot(b)
    #LocalVect = matrixVectorMult(LocalMatrix,b) # Local product
    #print(f"Rank {RANK} Localvector = {LocalVect[:10]}  from Localmatrix = {LocalMatrix} multiplying vector {b}")


    # Récupération des longueurs des morceaux de vecteurs calculer par processes
    sendcounts = np.array(COMM.gather(len(LocalVect), root=0))

    # Raccollement 
    if RANK==0:
        recvbuf = np.empty(sum(sendcounts) , dtype=np.double)
    else:
        recvbuf = None

    COMM.Gatherv(sendbuf=LocalVect , recvbuf=(recvbuf , sendcounts, MPI.DOUBLE ) , root=0 )
    m = MPI.Wtime()


    if RANK == 0 :
        print (f"Gatherarray : {recvbuf[:10]}")
        print("\n")

    # Comparaison avec le produit normal.
    if RANK == 0 :
        print(f"The result of A*b using parallel version is : {recvbuf[:10]} and time = {m-l}")

        u = MPI.Wtime()
        #X = matrixVectorMult(A,b)
        X = A.dot(b)
        v = MPI.Wtime()
        print("\n")
        print(f"The result of A*b using conventioal version is : {X[:10]} and time = {v-u}")
    return 0


#A = np.array([[1,3,4],[4,3,5],[1,1,1],[0,0,1]])
#b = [1,2,3]

SIZE = 1000
A = lil_matrix((SIZE, SIZE))
A[0, :100] = rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(rand(SIZE))
A = A.toarray()
b = rand(SIZE)

        

Parallel_Matrix_Vector_Mult(A,b)
