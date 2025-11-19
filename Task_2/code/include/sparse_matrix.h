#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matrix.h"

// Compressed Sparse Row (CSR) matrix structure
typedef struct {
    double *values;      // Non-zero values
    int *col_indices;    // Column indices of non-zero values
    int *row_pointers;   // Start index of each row in values array
    int rows;
    int cols;
    int nnz;            // Number of non-zero elements
} CSRMatrix;

// CSR matrix operations
CSRMatrix* csr_create(int rows, int cols, int nnz);
void csr_free(CSRMatrix* m);
CSRMatrix* csr_from_dense(Matrix* dense);
Matrix* csr_to_dense(CSRMatrix* csr);
CSRMatrix* csr_multiply(CSRMatrix* a, CSRMatrix* b);
void csr_print(CSRMatrix* m);

// Sparse matrix utilities
Matrix* matrix_create_sparse(int rows, int cols, double sparsity);
double matrix_calculate_sparsity(Matrix* m);

#endif // SPARSE_MATRIX_H