#include <stdio.h>
#include <stdlib.h>

void matrixMultiply(int n, int m, int k, int **A, int **B, int **C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i][j] = 0;
            for (int l = 0; l < m; l++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

int main() {
    int n = 2, m = 2, k = 2;

    // Allocate memory for matrices
    int **A = (int **)malloc(n * sizeof(int *));
    int **B = (int **)malloc(m * sizeof(int *));
    int **C = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        A[i] = (int *)malloc(m * sizeof(int));
        C[i] = (int *)malloc(k * sizeof(int));
    }
    for (int i = 0; i < m; i++) {
        B[i] = (int *)malloc(k * sizeof(int));
    }

    // Initialize matrices A and B with some values
    int counter = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i][j] = counter++;
        }
    }
    counter = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            B[i][j] = counter++;
        }
    }

    matrixMultiply(n, m, k, A, B, C);

    printf("Result matrix C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            printf("%d\t", C[i][j]);
        }
        printf("\n");
    }
    
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(C[i]);
    }
    for (int i = 0; i < m; i++) {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}