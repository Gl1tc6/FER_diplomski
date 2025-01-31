#include <stdio.h>
#include <stdlib.h>

// Funkcija za množenje matrica
void matrix_multiply(int **A, int **B, int **C, int n, int m, int k) {
    // Inicijalizacija rezultantne matrice na 0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i][j] = 0;
        }
    }
    
    // Implementacija matričnog množenja
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            for (int p = 0; p < m; p++) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
}

// Pomoćna funkcija za alokaciju 2D matrice
int** allocate_matrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}

// Pomoćna funkcija za oslobađanje memorije
void free_matrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Pomoćna funkcija za ispis matrice
void print_matrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int n = 2, m = 3, k = 2;  // Primjer dimenzija

    // Alokacija matrica
    int** A = allocate_matrix(n, m);
    int** B = allocate_matrix(m, k);
    int** C = allocate_matrix(n, k);

    // Primjer inicijalizacije prve matrice
    A[0][0] = 1; A[0][1] = 2; A[0][2] = 3;
    A[1][0] = 4; A[1][1] = 5; A[1][2] = 6;

    // Primjer inicijalizacije druge matrice
    B[0][0] = 7; B[0][1] = 8;
    B[1][0] = 9; B[1][1] = 10;
    B[2][0] = 11; B[2][1] = 12;

    // Množenje matrica
    matrix_multiply(A, B, C, n, m, k);

    printf("Matrica A:\n");
    print_matrix(A, n, m);
    printf("Matrica B:\n");
    print_matrix(B, m, k);

    // Ispis rezultata
    printf("C:\n");
    print_matrix(C, n, k);

    // Oslobađanje memorije
    free_matrix(A, n);
    free_matrix(B, m);
    free_matrix(C, n);

    return 0;
}