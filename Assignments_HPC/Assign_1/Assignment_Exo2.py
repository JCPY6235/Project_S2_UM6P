from mpi4py import MPI
#Communicator , Rank and s i z e
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
Tag = 1


Tag = 1


while True:
    if RANK == 0:
            value = int(input("Please tape an integer "))
            if value < 0:
                for r in range(1,SIZE):
                    COMM.send(value, dest = r, tag = Tag)
                break
            print(f"Process {0} got {value}")
            for r in range(1,SIZE):
                COMM.send(value, dest = r, tag = Tag)
    else:
        data = COMM.recv(source = 0, tag = Tag )
        if data <0:
             break
        print(f"Process {RANK} got {data}")

    

    


### Methode Bcast.
"""
if RANK == 0:
        value = int(input("Please tape an integer "))
        while value < 0:
             print("A value should be an integer")
             value = int(input("Please tape an integer "))
        
else:
    value = None
rcvbust = COMM.bcast(value, root = 0)

print(f"Process {RANK} got {rcvbust}")

"""




