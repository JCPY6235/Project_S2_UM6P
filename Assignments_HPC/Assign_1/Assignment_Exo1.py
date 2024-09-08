from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
#print("Hello World")
#print(f"Hello World {RANK} among {SIZE}")
if RANK == 2:
    print(f"Hello World {RANK} among {SIZE}")