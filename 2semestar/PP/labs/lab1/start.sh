#!/bin/bash

if [ -z "$1" ]; then
    read -p "Enter the number of processes: " num_processes
    set -- "$num_processes"
fi
mpicc -o filozofi zad1.c && mpirun -np $1 ./filozofi
#mpicc -g -o filozofi zad1.c && mpirun -n 3 xterm -e gdb ./filozofi

