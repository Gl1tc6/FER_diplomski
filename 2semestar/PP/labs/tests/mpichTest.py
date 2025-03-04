from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ARRAY_SIZE = 100

# Initialize the array on the root process (rank 0)
if rank == 0:
    array = np.arange(1, ARRAY_SIZE + 1, dtype=int)  # Array with values 1 to ARRAY_SIZE
else:
    array = np.empty(ARRAY_SIZE, dtype=int)

# Broadcast the array to all processes
comm.Bcast(array, root=0)

# Divide the work among processes
chunk_size = ARRAY_SIZE // size
start = rank * chunk_size
end = ARRAY_SIZE if rank == size - 1 else start + chunk_size

# Compute the local sum
local_sum = np.sum(array[start:end])

# Combine the local sums into a global sum
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Print the result on the root process
if rank == 0:
    print(f"Total sum of the array: {global_sum}")