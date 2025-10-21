/*
 * Matrix multiplication implementation in C
 */
#include <stdlib.h>
#include "matrix_multiply.h"

/**
 * Basic matrix multiplication algorithm O(n^3)
 */
double** matrix_multiply(double** A, double** B, int size) {
    double** C = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        C[i] = (double*)calloc(size, sizeof(double));
    }
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}