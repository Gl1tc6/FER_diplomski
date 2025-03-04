#!/usr/bin
# Use mpicc and mpirun from conda installation (MPICH) because system one
# (OpenMPI) isn't Python compatible!

# conda activate base

# For Python we use just mpirun
echo "Running Python parallel program using [$1] processes:"
mpirun -np $1 python mpichTest.py

echo "Running C parallel program using [$1] processes:"
# For C we need to compile using mpicc and then run with mpirun
mpicc -o mpichTest mpichTest.c 
mpirun -np $1 ./mpichTest 