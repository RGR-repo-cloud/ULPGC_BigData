#include "matrix.h"

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    m->rows = rows;
    m->cols = cols;
    
    // Allocate array of row pointers
    m->data = malloc(rows * sizeof(double*));
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    // Allocate contiguous memory for all matrix elements
    double* data_block = malloc(rows * cols * sizeof(double));
    if (!data_block) {
        free(m->data);
        free(m);
        return NULL;
    }
    
    // Set up row pointers
    for (int i = 0; i < rows; i++) {
        m->data[i] = data_block + i * cols;
    }
    
    return m;
}

void matrix_free(Matrix* m) {
    if (m) {
        if (m->data) {
            free(m->data[0]); // Free the contiguous data block
            free(m->data);    // Free the row pointers
        }
        free(m);
    }
}

void matrix_fill_random(Matrix* m, double min, double max) {
    if (!m) return;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            double random = (double)rand() / RAND_MAX;
            m->data[i][j] = min + random * (max - min);
        }
    }
}

void matrix_fill_zeros(Matrix* m) {
    if (!m) return;
    
    memset(m->data[0], 0, m->rows * m->cols * sizeof(double));
}

void matrix_fill_identity(Matrix* m) {
    if (!m || m->rows != m->cols) return;
    
    matrix_fill_zeros(m);
    for (int i = 0; i < m->rows; i++) {
        m->data[i][i] = 1.0;
    }
}

int matrix_verify_result(Matrix* a, Matrix* b, double tolerance) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) {
        return 0;
    }
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            if (fabs(a->data[i][j] - b->data[i][j]) > tolerance) {
                return 0;
            }
        }
    }
    return 1;
}

void matrix_print(Matrix* m) {
    if (!m) return;
    
    printf("Matrix %dx%d:\n", m->rows, m->cols);
    for (int i = 0; i < m->rows && i < 10; i++) { // Print max 10 rows
        for (int j = 0; j < m->cols && j < 10; j++) { // Print max 10 cols
            printf("%8.3f ", m->data[i][j]);
        }
        if (m->cols > 10) printf("...");
        printf("\n");
    }
    if (m->rows > 10) printf("...\n");
}

// Basic matrix multiplication O(n³)
Matrix* matrix_multiply_basic(Matrix* a, Matrix* b) {
    if (!a || !b || a->cols != b->rows) {
        return NULL;
    }
    
    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) return NULL;
    
    matrix_fill_zeros(result);
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            for (int k = 0; k < a->cols; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    
    return result;
}

// Matrix addition for Strassen algorithm
Matrix* matrix_add(Matrix* a, Matrix* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }
    
    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }
    
    return result;
}

// Matrix subtraction for Strassen algorithm
Matrix* matrix_subtract(Matrix* a, Matrix* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) {
        return NULL;
    }
    
    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    
    return result;
}

// Create submatrix for Strassen algorithm
Matrix* matrix_create_submatrix(Matrix* m, int start_row, int start_col, int size) {
    if (!m || start_row + size > m->rows || start_col + size > m->cols) {
        return NULL;
    }
    
    Matrix* sub = matrix_create(size, size);
    if (!sub) return NULL;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sub->data[i][j] = m->data[start_row + i][start_col + j];
        }
    }
    
    return sub;
}

// Copy submatrix back to main matrix
void matrix_copy_submatrix(Matrix* dest, Matrix* src, int dest_row, int dest_col) {
    if (!dest || !src) return;
    
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[dest_row + i][dest_col + j] = src->data[i][j];
        }
    }
}