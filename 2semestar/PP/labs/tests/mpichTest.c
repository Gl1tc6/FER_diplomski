#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100

int main(int argc, char** argv) {
    int rank, size;
    int local_sum = 0, global_sum = 0;
    int* array = (int*)malloc(ARRAY_SIZE * sizeof(int));

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the array on the root process (rank 0)
    if (rank == 0) {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = i + 1; // Fill array with values 1 to ARRAY_SIZE
        }
    }

    // Broadcast the array to all processes
    MPI_Bcast(array, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide the work among processes
    int chunk_size = ARRAY_SIZE / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? ARRAY_SIZE : start + chunk_size;

    // Compute the local sum
    for (int i = start; i < end; i++) {
        local_sum += array[i];
    }

    // Combine the local sums into a global sum
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print the result on the root process
    if (rank == 0) {
        printf("Total sum of the array: %d\n", global_sum);
    }

    // Clean up
    free(array);
    MPI_Finalize();
    return 0;
}