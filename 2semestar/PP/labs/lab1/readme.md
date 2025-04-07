# Distributed Philosophers Problem

## Prerequisites
- MPICH library installed
- GCC compiler

## Compilation
```bash
mpicc -o filozofi zad1.c
```

## Running
```bash
mpirun -np 5 ./filozofi
```

## Notes
- The number of processes (`-np`) should be at least 2
- Adjust the number of processes as needed
- The implementation follows the distributed philosophers problem algorithm described in K.M. Chandy, J. Misra's paper

## Algorithm Details
- Philosophers alternate between thinking and eating
- Forks are managed by processes
- Forks start dirty and are cleaned when used
- Processes communicate via message passing
- The first process (rank 0) manages fork requests
