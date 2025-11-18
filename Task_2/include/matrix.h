#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// Matrix structure for dense matrices
typedef struct {
    double **data;
    int rows;
    int cols;
} Matrix;

// Basic matrix operations
Matrix* matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
void matrix_fill_random(Matrix* m, double min, double max);
void matrix_fill_zeros(Matrix* m);
void matrix_fill_identity(Matrix* m);
int matrix_verify_result(Matrix* a, Matrix* b, double tolerance);
void matrix_print(Matrix* m);

// Matrix multiplication algorithms
Matrix* matrix_multiply_basic(Matrix* a, Matrix* b);
Matrix* matrix_multiply_strassen(Matrix* a, Matrix* b);
Matrix* matrix_multiply_loop_unroll(Matrix* a, Matrix* b);
Matrix* matrix_multiply_cache_block(Matrix* a, Matrix* b, int block_size);

// Utility functions
Matrix* matrix_add(Matrix* a, Matrix* b);
Matrix* matrix_subtract(Matrix* a, Matrix* b);
Matrix* matrix_create_submatrix(Matrix* m, int start_row, int start_col, int size);
void matrix_copy_submatrix(Matrix* dest, Matrix* src, int dest_row, int dest_col);

#endif // MATRIX_H