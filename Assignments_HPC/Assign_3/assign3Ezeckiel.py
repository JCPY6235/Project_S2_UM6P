"""

/*
 *   Solving the Poisson's equation discretized on the [0,1]x[0,1] domain
 *   using the finite difference method and a Jacobi's iterative solver.
 *
 *   Delta u = f(x,y)= 2*(x*x-x+y*y -y)
 *   u equal 0 on the boudaries
 *   The exact solution is u = x*y*(x-1)*(y-1)
 *
 *   The u value is :
 *    coef(1) = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
 *    coef(2) = 1./(hx*hx)
 *    coef(3) = 1./(hy*hy)
 *
 *    u(i,j)(n+1)= coef(1) * (  coef(2)*(u(i+1,j)+u(i-1,j)) &
 *               + coef(3)*(u(i,j+1)+u(i,j-1)) - f(i,j))
 *
 *   ntx and nty are the total number of interior points along x and y, respectivly.
 * 
 *   hx is the grid spacing along x and hy is the grid spacing along y.
 *    hx = 1./(ntx+1)
 *    hy = 1./(nty+1)
 *
 *   On each process, we need to:
 *   1) Split up the domain
 *   2) Find our 4 neighbors
 *   3) Exchange the interface points
 *   4) Calculate u
 *
 *   @author: kissami
 */


"""
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
#from numba import njit
from utils import compute_dims




comm = MPI.COMM_WORLD
nb_procs = comm.Get_size()
rank = comm.Get_rank()

nb_neighbours = 4
N = 0
E = 1
S = 2
W = 3

neighbour = np.zeros(nb_neighbours, dtype=np.int8)

# Number of interior points of the initial domain
# along x

ntx = 6
# along y
nty = 4


# Number of points of the initial domain, including the boundary points
# along x
Nx = ntx+2
# along x
Ny = nty+2

# Number of interior points of the initial domain along x and y axis
npoints  =  [ntx, nty]
p1 = [2,2]
P1 = [False, False]
reorder = True


coef = np.zeros(3)
''' Grid spacing '''
hx = 1/(ntx+1.);
hy = 1/(nty+1.);

''' Equation Coefficients '''
coef[0] = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy);
coef[1] = 1./(hx*hx);
coef[2] = 1./(hy*hy);


def distribute_points(N, p):
    """
    Distribute points among a given number of elements.
    """
    
    eq = N // p  # Calculate equal distribution among elements
    remainder = N % p  # Calculate remainder after equal distribution
    arr = np.full(p, eq,dtype='i')  # Initialize array with equal distribution

    #for i in range(remainder,0,-1):
    for i in range(remainder):
        arr[i] += 1

    return arr


def create_2d_cart(npoints, p1, P1, reorder):
    
    # Store input arguments                                                                                                                                                                                                                                               
    npts    = tuple(npoints)
    pads    = tuple(p1)
    periods = tuple(P1)
    reorder = reorder
    
    nprocs, _ = compute_dims(nb_procs, npts, pads )
    
    dims = nprocs
    
    if (rank == 0):
        print("Execution poisson with",nb_procs," MPI processes\n"
               "Size of the (original)domain : ntx=",npoints[0], " nty=",npoints[1],"\n"
              
              ,dims[0],"processes along x and", dims[1],"processes along y\n"
               #"Dimension for the topology :",dims[0]," along x", dims[1]," along y\n"
               "-----------------------------------------")  
    
    '''
    * Creation of the Cartesian topology
    '''
    cart2d=comm.Create_cart(dims=nprocs,reorder=reorder,periods=periods)
    
    return dims, cart2d

def create_2dCoords(cart2d, npoints, dims):

    ''' Create 2d coordinates of each process'''
    
    #print(dims)
    
    width,height=npoints
    
    nprocs=dims
    
    
    
    #counts_x=np.empty(nprocs[0],dtype='i')
    #counts_y=np.empty(nprocs[1],dtype='i')
    
    #displx=np.empty(nprocs[0],dtype='i')
    #disply=np.empty(nprocs[1],dtype='i')
    
    nprocs_x,nprocs_y=nprocs

    if rank==0:
        pass
        


        #counts_x=distribute_points(width,nprocs_x)

        #counts_y=distribute_points(height,nprocs_y)
        
        #displx = [sum(cols_nums[:p]) for p in range(nprocs_x)]
        #disply = [sum(rows_nums[:p]) for p in range(nprocs_y)]
        #displx = np.array(displx)
        #disply = np.array(disply)
        #print('diaplacements along x:{} along y:{}'.format(displx,disply))
        

    # Send the numbers of cells along x and y axes to each process
    #comm.Bcast([counts_x, nprocs[0],MPI.INT],root=0)
    #comm.Bcast([counts_y, nprocs[1],MPI.INT],root=0)
    
    #displx = [sum(counts_x[:p]) for p in range(nprocs_x)]
    #disply = [sum(counts_y[:p]) for p in range(nprocs_y)]
    #displx = np.array(displx)
    #disply = np.array(disply)
    
    if rank==0:
        pass
        #print('displacements along x:{} along y:{}'.format(displx,disply))
    
    # Send the displacements of cells along x and y axes to each process
    #comm.Bcast([displx, nprocs[0],MPI.INT],root=0)
    #comm.Bcast([disply, nprocs[1],MPI.INT],root=0)
    
    
    


    #if rank==0:
        #pass
        #print('Number of columns:{}\nNumber of rows:{}'.format(cols_nums,rows_nums))
        #print('Number of columns:{}\nNumber of rows:{}'.format(cols_nums,rows_nums))
    

    i,j=cart2d.Get_coords(rank)
    

    #print(' Coords of {}: ({},{})'.format(rank,i,j))
    

    
    #sx=displx[i]
    #ex=sx+counts_x[i]

    #sy=disply[j]
    #ey=sy+counts_y[j]
    
    sx=int((i*npoints[0])/dims[0]+1)
    ex=int(((i+1)*npoints[0])/dims[0])
    
    sy=int((j*npoints[1])/dims[1]+1)
    ey=int(((j+1)*npoints[1])/dims[1])
    
    

    print("Rank in the topology :",rank," Local Grid Index :", sx, " to ",ex," along x, ",
          sy, " to", ey," along y")
    
    return sx, ex, sy, ey

def create_neighbours(cart2d):

    ''' Get my northern and southern neighbours '''
    low, high = cart2d.Shift(direction=1, disp=1)
    
    
    

    ''' Get my western and eastern neighbours '''
    left, right = cart2d.Shift(direction=0, disp=1)
    neighbour=[high,left,low,right]

    
    print("Process", rank," neighbour: N", neighbour[N]," E",neighbour[E] ," S ",neighbour[S]," W",neighbour[W])
    
    return neighbour
'''Creation of the derived datatypes needed to exchange points with neighbours'''
def create_derived_type(sx, ex, sy, ey):
    '''Creation of the type_line derived datatype to exchange points
     with northern to southern neighbours '''
    countx=ex-sx+1
    #type_ligne= MPI.DOUBLE.Create_contiguous(countx)
    #type_ligne.Commit()
    
    '''Creation of the type_column derived datatype to exchange points
     with western to eastern neighbours '''
    county=ey-sy+1
    bloclen=1
    #stride=countx
    #stride=ex-sx+3
    stride=countx+2
    #type_column=MPI.DOUBLE.Create_vector(county,bloclen,stride)
    type_column= MPI.DOUBLE.Create_contiguous(ey-sy+3)
    type_column.Commit()
    type_ligne=MPI.DOUBLE.Create_vector(ex-sx+3,1,ey-sy+3)
    type_ligne.Commit()
    
    return type_ligne, type_column

''' Exchange the points at the interface '''
def communications(u, sx, ex, sy, ey, type_column, type_ligne):  
     
    ''' Send to neighbour N and receive from neighbour S '''
    comm.Sendrecv(
        sendbuf=[ u ,1,type_ligne],
        dest=neighbour[N],
        recvbuf=[u[(ey-sy+2):] ,1,type_ligne],
        source=neighbour[S]
        )
   

    ''' Send to neighbour S and receive from neighbour N '''
    comm.Sendrecv(
        sendbuf=[ u[(ey-sy+1):] ,1,type_ligne],
        dest=neighbour[S],
        recvbuf=[ u ,1,type_ligne],
        source=neighbour[N]
        )
    

    ''' Send to neighbour W and receive from neighbour E '''
    comm.Sendrecv(
        sendbuf=[ u[ey-sy+3:] ,1,type_column],
        dest=neighbour[W],
        recvbuf=[ u[(ey-sy+3)*(ex-sx+2):] ,1,type_column],
        source=neighbour[E]
        )
    


    ''' Send to neighbour E  and receive from neighbour W '''
    comm.Sendrecv(
        sendbuf=[ u[(ey-sy+3)*(ex-sx+1):]  ,1,type_column],
        dest=neighbour[E],
        recvbuf=[ u,1,type_column],
        source=neighbour[W]
        )

'''
 * IDX(i, j) : indice de l'element i, j dans le tableau u
 * sx-1 <= i <= ex+1
 * sy-1 <= j <= ey+1
'''

def IDX(i, j):
    return ( ((i)-(sx-1))*(ey-sy+3) + (j)-(sy-1) )

def initialization(sx, ex, sy, ey):
    ''' Grid spacing in each dimension'''
    ''' Solution u and u_new at the n and n+1 iterations '''
    SIZE = (ex-sx+3) * (ey-sy+3)
    
    u    = np.zeros(SIZE)      
                
                
    u_new   = np.zeros(SIZE)
    f       = np.zeros(SIZE)
    u_exact = np.zeros(SIZE)
    
    '''Initialition of rhs f and exact soluction '''
    
    for i in range(sx, ex ):
        for j in range(sy, ey):
            x = i * hx
            y = j * hy
            f[IDX(i, j)] = 2 * (x * x - x + y * y - y)
            u_exact[IDX(i, j)] = x * y * (x - 1) * (y - 1)
    
        

    
    return u, u_new, u_exact, f

def computation(u, u_new):
    
    ''' Compute the new value of u using 
    
    '''
    for i in range(sx, ex+1):
        for j in range(sy, ey+1):
            u_new[IDX(i, j)] = coef[0] * (coef[1] * (u[IDX(i+1, j)] + u[IDX(i-1, j)]) +
                                          coef[2] * (u[IDX(i, j+1)] + u[IDX(i, j-1)]) - f[IDX(i, j)])

 
def output_results(u, u_exact):
    
    print("Exact Solution u_exact - Computed Solution u - difference")
    for itery in range(sy, ey+1, 1):
        print(u_exact[IDX(1, itery)], '-', u[IDX(1, itery)], u_exact[IDX(1, itery)]-u[IDX(1, itery)] );

''' Calcul for the global error (maximum of the locals errors) '''
def global_error(u, u_new):
   
    local_error = 0
     
    for iterx in range(sx, ex+1, 1):
        for itery in range(sy, ey+1, 1):
            temp = np.fabs( u[IDX(iterx, itery)] - u_new[IDX(iterx, itery)]  )
            if local_error < temp:
                local_error = temp;
    
    return local_error


def plot_2d(f):

    f = np.reshape(f, (ex-sx+3, ey-sy+3))
    
    x = np.linspace(0, 1, ey-sy+3)
    y = np.linspace(0, 1, ex-sx+3)
    
    fig = plt.figure(figsize=(7, 5), dpi=100)
    #ax = fig.gca(projection='3d')
    
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)      

    ax.plot_surface(X, Y, f, cmap=cm.viridis)
    
    plt.show()

dims, cart2d   = create_2d_cart(npoints, p1, P1, reorder)
neighbour      = create_neighbours(cart2d)

sx, ex, sy, ey = create_2dCoords(cart2d, npoints, dims)

type_ligne, type_column = create_derived_type(sx, ex, sy, ey)
u, u_new, u_exact, f             = initialization(sx, ex, sy, ey)

''' Time stepping '''
it = 0
convergence = False
it_max = 100000
eps = 2.e-16

''' Elapsed time '''
t1 = MPI.Wtime()

#import sys; sys.exit()
while (not(convergence) and (it < it_max)):
    it = it+1;

    temp = u.copy() 
    u = u_new.copy() 
    u_new = temp.copy()
    
    ''' Exchange of the interfaces at the n iteration '''
    communications(u, sx, ex, sy, ey, type_column, type_ligne)
   
    ''' Computation of u at the n+1 iteration '''
    
    computation(u, u_new)
    
    ''' Computation of the global error '''
    local_error = global_error(u, u_new);
    diffnorm = comm.allreduce(np.array(local_error), op=MPI.MAX )   
   
    ''' Stop if we obtained the machine precision '''
    convergence = (diffnorm < eps)
    
    ''' Print diffnorm for process 0 '''
    if ((rank == 0) and ((it % 100) == 0)):
        print("Iteration", it, " global_error = ", diffnorm);
        
''' Elapsed time '''
t2 = MPI.Wtime()

if (rank == 0):
    ''' Print convergence time for process 0 '''
    print("Convergence after",it, 'iterations in', t2-t1,'secs')

    ''' Compare to the exact solution on process 0 '''
    output_results(u, u_exact)
    
    plot_2d(f)
