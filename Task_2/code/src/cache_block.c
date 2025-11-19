#include "matrix.h"

// Cache-blocked matrix multiplication
Matrix* matrix_multiply_cache_block(Matrix* a, Matrix* b, int block_size) {
    if (!a || !b || a->cols != b->rows) return NULL;
    
    if (block_size <= 0) block_size = 64;
    
    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) return NULL;
    matrix_fill_zeros(result);
    
    // Cache blocking: process in blocks to improve locality
    for (int i = 0; i < a->rows; i += block_size) {
        for (int j = 0; j < b->cols; j += block_size) {
            for (int k = 0; k < a->cols; k += block_size) {
                
                // Process current block
                int i_max = (i + block_size < a->rows) ? i + block_size : a->rows;
                int j_max = (j + block_size < b->cols) ? j + block_size : b->cols;
                int k_max = (k + block_size < a->cols) ? k + block_size : a->cols;
                
                for (int ii = i; ii < i_max; ii++) {
                    for (int jj = j; jj < j_max; jj++) {
                        for (int kk = k; kk < k_max; kk++) {
                            result->data[ii][jj] += a->data[ii][kk] * b->data[kk][jj];
                        }
                    }
                }
            }
        }
    }
    
    return result;
}