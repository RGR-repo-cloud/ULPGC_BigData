/*
 * Matrix Utilities for C
 */
#include <stdio.h>
#include <stdlib.h>
#include "matrix_utils.h"

/**
 * Create a random matrix of given size
 */
double** create_random_matrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
        for (int j = 0; j < size; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

/**
 * Create a zero matrix of given size
 */
double** create_zero_matrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)calloc(size, sizeof(double));
    }
    return matrix;
}

/**
 * Free matrix memory
 */
void free_matrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}