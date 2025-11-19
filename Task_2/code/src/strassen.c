#include "matrix.h"

#define STRASSEN_THRESHOLD 64

// Strassen algorithm - optimized with threshold for practical performance
Matrix* matrix_multiply_strassen(Matrix* a, Matrix* b) {
    if (!a || !b || a->cols != b->rows) return NULL;
    
    int n = a->rows;
    // Use basic multiplication for small matrices or non-square/non-power-of-2 matrices
    if (n <= STRASSEN_THRESHOLD || n != a->cols || n != b->rows || n != b->cols || (n & (n - 1)) != 0) {
        return matrix_multiply_basic(a, b);
    }
    
    int half = n / 2;
    
    // Create submatrices
    Matrix* a11 = matrix_create_submatrix(a, 0, 0, half);
    Matrix* a12 = matrix_create_submatrix(a, 0, half, half);
    Matrix* a21 = matrix_create_submatrix(a, half, 0, half);
    Matrix* a22 = matrix_create_submatrix(a, half, half, half);
    Matrix* b11 = matrix_create_submatrix(b, 0, 0, half);
    Matrix* b12 = matrix_create_submatrix(b, 0, half, half);
    Matrix* b21 = matrix_create_submatrix(b, half, 0, half);
    Matrix* b22 = matrix_create_submatrix(b, half, half, half);
    
    // Compute 7 Strassen products with temporary matrices
    Matrix *t1, *t2;
    
    t1 = matrix_add(a11, a22); t2 = matrix_add(b11, b22);
    Matrix* m1 = matrix_multiply_strassen(t1, t2);
    matrix_free(t1); matrix_free(t2);
    
    t1 = matrix_add(a21, a22);
    Matrix* m2 = matrix_multiply_strassen(t1, b11);
    matrix_free(t1);
    
    t1 = matrix_subtract(b12, b22);
    Matrix* m3 = matrix_multiply_strassen(a11, t1);
    matrix_free(t1);
    
    t1 = matrix_subtract(b21, b11);
    Matrix* m4 = matrix_multiply_strassen(a22, t1);
    matrix_free(t1);
    
    t1 = matrix_add(a11, a12);
    Matrix* m5 = matrix_multiply_strassen(t1, b22);
    matrix_free(t1);
    
    t1 = matrix_subtract(a21, a11); t2 = matrix_add(b11, b12);
    Matrix* m6 = matrix_multiply_strassen(t1, t2);
    matrix_free(t1); matrix_free(t2);
    
    t1 = matrix_subtract(a12, a22); t2 = matrix_add(b21, b22);
    Matrix* m7 = matrix_multiply_strassen(t1, t2);
    matrix_free(t1); matrix_free(t2);
    
    matrix_free(a11); matrix_free(a12); matrix_free(a21); matrix_free(a22);
    matrix_free(b11); matrix_free(b12); matrix_free(b21); matrix_free(b22);
    
    // Compute result quadrants
    t1 = matrix_add(m1, m4); t2 = matrix_subtract(t1, m5);
    Matrix* c11 = matrix_add(t2, m7);
    matrix_free(t1); matrix_free(t2);
    
    Matrix* c12 = matrix_add(m3, m5);
    Matrix* c21 = matrix_add(m2, m4);
    
    t1 = matrix_subtract(m1, m2); t2 = matrix_add(t1, m3);
    Matrix* c22 = matrix_add(t2, m6);
    matrix_free(t1); matrix_free(t2);
    
    matrix_free(m1); matrix_free(m2); matrix_free(m3); matrix_free(m4);
    matrix_free(m5); matrix_free(m6); matrix_free(m7);
    
    // Assemble final result
    Matrix* result = matrix_create(n, n);
    matrix_copy_submatrix(result, c11, 0, 0);
    matrix_copy_submatrix(result, c12, 0, half);
    matrix_copy_submatrix(result, c21, half, 0);
    matrix_copy_submatrix(result, c22, half, half);
    
    matrix_free(c11); matrix_free(c12); matrix_free(c21); matrix_free(c22);
    return result;
}