#include "benchmark.h"
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>

// Simple benchmark for sparse matrix multiplication
PerformanceResult benchmark_sparse_algorithm(CSRMatrix* a, CSRMatrix* b, const char* name) {
    if (!a || !b) {
        return (PerformanceResult){0.0, 0, 0};
    }
    
    if (strlen(name) > 0) {
        printf("Testing %s (sparse %dx%d, nnz: %d+%d)... ", name, a->rows, b->cols, a->nnz, b->nnz);
        fflush(stdout);
    }
    
    malloc_trim(0);
    long mem_before = get_memory_usage();
    
    double time_start = get_wall_time();
    CSRMatrix* product = csr_multiply(a, b);
    double time_end = get_wall_time();
    
    long mem_after = get_memory_usage();
    PerformanceResult result = measure_performance(product, time_start, time_end, mem_before, mem_after, a->rows, name);
    
    csr_free(product);
    return result;
}

// Multi-run benchmark for sparse algorithms
PerformanceStats benchmark_sparse_algorithm_stats(CSRMatrix* a, CSRMatrix* b, const char* name, int num_runs) {
    PerformanceResult* results = malloc(num_runs * sizeof(PerformanceResult));
    if (!results) return (PerformanceStats){0};
    
    printf("%s ", name);
    fflush(stdout);
    
    for (int i = 0; i < num_runs; i++) {
        results[i] = benchmark_sparse_algorithm(a, b, "");
        printf(".");
        fflush(stdout);
    }
    
    PerformanceStats stats = calculate_stats(results, num_runs);
    printf(" %.3fs ", stats.avg_time);
    
    free(results);
    return stats;
}

// Simple sparsity analysis
void run_sparsity_analysis(int matrix_size, double min_sparsity __attribute__((unused)), double max_sparsity __attribute__((unused)), int steps __attribute__((unused))) {
    printf("\n=== SPARSE MATRIX ANALYSIS ===\n");
    printf("Matrix size: %dx%d\n", matrix_size, matrix_size);
    
    double sparsity_levels[] = {0.5, 0.7, 0.9, 0.95};
    int num_levels = sizeof(sparsity_levels) / sizeof(sparsity_levels[0]);
    const int num_runs = 3;
    
    for (int i = 0; i < num_levels; i++) {
        double sparsity = sparsity_levels[i];
        printf("\n--- %.0f%% Sparse ---\n", sparsity * 100);
        
        // Create and test matrices
        Matrix* a_dense = matrix_create_sparse(matrix_size, matrix_size, sparsity);
        Matrix* b_dense = matrix_create_sparse(matrix_size, matrix_size, sparsity);
        
        if (!a_dense || !b_dense) {
            printf("Failed to create sparse matrices\n");
            matrix_free(a_dense);
            matrix_free(b_dense);
            continue;
        }
        
        CSRMatrix* a_sparse = csr_from_dense(a_dense);
        CSRMatrix* b_sparse = csr_from_dense(b_dense);
        
        // Collect individual run results for detailed export
        PerformanceResult* dense_results = malloc(num_runs * sizeof(PerformanceResult));
        PerformanceResult* sparse_results = malloc(num_runs * sizeof(PerformanceResult));
        
        if (!dense_results || !sparse_results) {
            printf("Failed to allocate memory for detailed results\n");
            free(dense_results);
            free(sparse_results);
            matrix_free(a_dense);
            matrix_free(b_dense);
            csr_free(a_sparse);
            csr_free(b_sparse);
            continue;
        }
        
        // Collect individual run results
        for (int run = 0; run < num_runs; run++) {
            dense_results[run] = benchmark_algorithm(matrix_multiply_basic, a_dense, b_dense, "");
            sparse_results[run] = benchmark_sparse_algorithm(a_sparse, b_sparse, "");
        }
        
        // Calculate statistics from individual results
        PerformanceStats dense_result = calculate_stats(dense_results, num_runs);
        PerformanceStats sparse_result = calculate_stats(sparse_results, num_runs);
        
        printf("Dense ... %.3fs Sparse ... %.3fs ", dense_result.avg_time, sparse_result.avg_time);
        
        // Calculate metrics and export
        int storage_dense = matrix_size * matrix_size;
        int storage_sparse = a_sparse->nnz + b_sparse->nnz;
        double speedup = (dense_result.avg_time > 0) ? dense_result.avg_time / sparse_result.avg_time : 1.0;
        double memory_savings = (double)(storage_dense - storage_sparse) / storage_dense * 100.0;
        
        // Export both statistical and detailed results
        export_sparse_results_csv("sparse_results.csv", matrix_size, sparsity * 100, 
                                 dense_result, sparse_result, storage_dense, storage_sparse, (i == 0));
        export_detailed_sparse_results_csv("sparse_results.csv", matrix_size, sparsity * 100,
                                          dense_results, sparse_results, num_runs, (i == 0));
        
        printf("✓ %.0f%% sparse: %.1fx speedup, %.1f%% storage saved\n", 
               sparsity * 100, speedup, memory_savings);
        
        // Cleanup
        free(dense_results);
        free(sparse_results);
        matrix_free(a_dense);
        matrix_free(b_dense);
        csr_free(a_sparse);
        csr_free(b_sparse);
    }
    
    printf("\n✓ Sparse matrix analysis complete.\n");
}

