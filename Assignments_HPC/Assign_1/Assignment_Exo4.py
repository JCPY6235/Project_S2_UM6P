from mpi4py import MPI
#Communicator , Rank and s i z e
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
#Tag = 1


if RANK == 0:
    data = int(input("Please tape an integer "))
    while data < 0:
        print("A value should be an integer")
        data = int(input("Please tape an integer "))
    print(f"I Process {0} got {data} from user JC")
    COMM.send(data, dest = RANK+1, tag = 0)

for i in range(1,SIZE-1):
     if RANK == i:
        data = COMM.recv(source = RANK-1, tag = 0)
        print(f"I Process {RANK} got {data} from Process {RANK-1}" )
        COMM.send(data+i, dest = i+1, tag = 0)

if RANK == SIZE-1:
    data = COMM.recv(source = RANK-1, tag = 0)
    print(f"I Process {RANK} got {data} from Process {RANK-1}")
          


    
           
    
