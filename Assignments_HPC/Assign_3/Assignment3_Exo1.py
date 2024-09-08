from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def grid_initializer(rows, cols):
   return np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])


def Update_state(grid):
   rows, cols = grid.shape
   new_grid = np.zeros_like(grid)


   for i in range(1, rows - 1):
       for j in range(1, cols - 1):
           alive_neighbors = np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]
           
           if grid[i, j] == 1 and (alive_neighbors < 2 or alive_neighbors > 3):
               new_grid[i, j] = 0
           elif grid[i, j] == 0 and alive_neighbors == 3:
               new_grid[i, j] = 1
           else:
               new_grid[i, j] = grid[i, j]
   
   return new_grid


def grid_plotter(grid, title):
   plt.imshow(grid, cmap='binary')
   plt.title(title)
   plt.show()




# Ensure the number of processes is smaller than or equal to the number of rows
rows = size 
cols = 10


# Create a Cartesian communicator
dims = [size, 1]
periods = [False, False]
cart_comm = comm.Create_cart(dims, periods)


sub_rows = rows // size
sub_grid = np.zeros((sub_rows + 2, cols), dtype=int)

# Initialize and scatter the global grid among processes
if rank == 0:
global_grid = grid_initializer(rows, cols)
else:
global_grid = None

global_grid = comm.bcast(global_grid, root=0)
local_grid = np.zeros((sub_rows + 2, cols), dtype=int)
comm.Scatter(global_grid[:sub_rows * size, :], local_grid[1:-1, :])


# Determine neighboring processes
north, south = cart_comm.Shift(0, 1)


# Game simulation
generations = 5
for gen in range(generations):
    comm.Sendrecv([local_grid[-2, :], MPI.INT], dest=south, source=north, recvbuf=local_grid[0, :])
    comm.Sendrecv([local_grid[1, :], MPI.INT], dest=north, source=south, recvbuf=local_grid[-1, :])
    local_grid = Update_state(local_grid)


global_grid = comm.gather(local_grid[1:-1, :], root=0)


if rank == 0:
   print("Generation:", gen + 1)
   print(np.vstack(global_grid))
   grid_plotter(np.vstack(global_grid), f"Generation {gen + 1}")





