#include "sparse_matrix.h"

// Create CSR matrix
CSRMatrix* csr_create(int rows, int cols, int nnz) {
    CSRMatrix* csr = malloc(sizeof(CSRMatrix));
    if (!csr) return NULL;
    
    csr->rows = rows;
    csr->cols = cols;
    csr->nnz = nnz;
    csr->values = malloc(nnz * sizeof(double));
    csr->col_indices = malloc(nnz * sizeof(int));
    csr->row_pointers = calloc(rows + 1, sizeof(int));
    
    if (!csr->values || !csr->col_indices || !csr->row_pointers) {
        csr_free(csr);
        return NULL;
    }
    return csr;
}

void csr_free(CSRMatrix* csr) {
    if (csr) {
        free(csr->values);
        free(csr->col_indices);
        free(csr->row_pointers);
        free(csr);
    }
}

CSRMatrix* csr_from_dense(Matrix* dense) {
    if (!dense) return NULL;
    
    // Count non-zeros
    int nnz = 0;
    for (int i = 0; i < dense->rows; i++) {
        for (int j = 0; j < dense->cols; j++) {
            if (fabs(dense->data[i][j]) > 1e-12) nnz++;
        }
    }
    
    CSRMatrix* csr = csr_create(dense->rows, dense->cols, nnz);
    if (!csr) return NULL;
    
    // Fill CSR structure
    int idx = 0;
    for (int i = 0; i < dense->rows; i++) {
        csr->row_pointers[i] = idx;
        for (int j = 0; j < dense->cols; j++) {
            if (fabs(dense->data[i][j]) > 1e-12) {
                csr->values[idx] = dense->data[i][j];
                csr->col_indices[idx] = j;
                idx++;
            }
        }
    }
    csr->row_pointers[dense->rows] = nnz;
    return csr;
}

Matrix* csr_to_dense(CSRMatrix* csr) {
    if (!csr) return NULL;
    
    Matrix* dense = matrix_create(csr->rows, csr->cols);
    if (!dense) return NULL;
    matrix_fill_zeros(dense);
    
    for (int i = 0; i < csr->rows; i++) {
        for (int j = csr->row_pointers[i]; j < csr->row_pointers[i + 1]; j++) {
            dense->data[i][csr->col_indices[j]] = csr->values[j];
        }
    }
    return dense;
}

CSRMatrix* csr_multiply(CSRMatrix* a, CSRMatrix* b) {
    if (!a || !b || a->cols != b->rows) return NULL;
    
    Matrix* dense_result = matrix_create(a->rows, b->cols);
    if (!dense_result) return NULL;
    matrix_fill_zeros(dense_result);
    
    // CSR multiplication: only process non-zero elements
    for (int i = 0; i < a->rows; i++) {
        for (int j = a->row_pointers[i]; j < a->row_pointers[i + 1]; j++) {
            int a_col = a->col_indices[j];
            double a_val = a->values[j];
            
            for (int k = b->row_pointers[a_col]; k < b->row_pointers[a_col + 1]; k++) {
                dense_result->data[i][b->col_indices[k]] += a_val * b->values[k];
            }
        }
    }
    
    CSRMatrix* result = csr_from_dense(dense_result);
    matrix_free(dense_result);
    return result;
}

void csr_print(CSRMatrix* csr) {
    if (!csr) return;
    printf("CSR Matrix %dx%d (nnz=%d)\n", csr->rows, csr->cols, csr->nnz);
}

Matrix* matrix_create_sparse(int rows, int cols, double sparsity) {
    Matrix* m = matrix_create(rows, cols);
    if (!m) return NULL;
    
    matrix_fill_random(m, -10.0, 10.0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((double)rand() / RAND_MAX < sparsity) {
                m->data[i][j] = 0.0;
            }
        }
    }
    return m;
}

double matrix_calculate_sparsity(Matrix* m) {
    if (!m) return 0.0;
    
    int zero_elements = 0;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (fabs(m->data[i][j]) < 1e-12) zero_elements++;
        }
    }
    return (double)zero_elements / (m->rows * m->cols);
}