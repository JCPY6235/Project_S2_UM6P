from mpi4py import MPI

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
value = 0

N = 10 # Number of iterations

for i in range(N):
    if RANK == 0:
        COMM.send(value, dest=1)
        value = COMM.recv(source=1)
        print(f"Step {i}:I, process {RANK}, I received {value} from the process 1.")
        
    if RANK == 1:
        value = COMM.recv(source=0)
        print(f"Step {i}:I, process {RANK}, I received {value} from the process 0.")
        value += 1
        COMM.send(value, dest=0)