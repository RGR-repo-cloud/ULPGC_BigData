#include "matrix.h"

// Loop unrolling
Matrix* matrix_multiply_loop_unroll(Matrix* a, Matrix* b) {
    if (!a || !b || a->cols != b->rows) return NULL;
    
    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) return NULL;
    matrix_fill_zeros(result);
    
    // Unroll inner loop by 4 for reduced overhead
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            
            int k;
            for (k = 0; k <= a->cols - 4; k += 4) {
                sum += a->data[i][k] * b->data[k][j] +
                       a->data[i][k+1] * b->data[k+1][j] +
                       a->data[i][k+2] * b->data[k+2][j] +
                       a->data[i][k+3] * b->data[k+3][j];
            }
            
            for (; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            
            result->data[i][j] = sum;
        }
    }
    
    return result;
}