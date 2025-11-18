#include "benchmark.h"
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>

// Simple benchmark for matrix multiplication algorithms
PerformanceResult benchmark_algorithm(Matrix* (*multiply_func)(Matrix*, Matrix*), 
                                    Matrix* a, Matrix* b, const char* name) {
    if (!multiply_func || !a || !b) {
        return (PerformanceResult){0.0, 0, 0};
    }
    
    if (strlen(name) > 0) {
        printf("Testing %s (%dx%d)... ", name, a->rows, b->cols);
        fflush(stdout);
    }
    
    malloc_trim(0);
    long mem_before = get_memory_usage();
    
    double time_start = get_wall_time();
    Matrix* product = multiply_func(a, b);
    double time_end = get_wall_time();
    
    // Ensure result matrix memory is committed
    if (product && product->data && product->rows > 0) {
        volatile double temp = product->data[0][0];
        if (product->rows > 1) temp += product->data[product->rows-1][product->cols-1];
        (void)temp;
    }
    
    long mem_after = get_memory_usage();
    PerformanceResult result = measure_performance(product, time_start, time_end, mem_before, mem_after, a->rows, name);
    
    matrix_free(product);
    return result;
}

// Simple benchmark for cache-blocked matrix multiplication
PerformanceResult benchmark_cache_algorithm(Matrix* (*multiply_func)(Matrix*, Matrix*, int), 
                                          Matrix* a, Matrix* b, int block_size, const char* name) {
    if (!multiply_func || !a || !b) {
        return (PerformanceResult){0.0, 0, 0};
    }
    
    if (strlen(name) > 0) {
        printf("Testing %s (%dx%d)... ", name, a->rows, b->cols);
        fflush(stdout);
    }
    
    malloc_trim(0);
    long mem_before = get_memory_usage();
    
    double time_start = get_wall_time();
    Matrix* product = multiply_func(a, b, block_size);
    double time_end = get_wall_time();
    
    // Ensure result matrix memory is committed
    if (product && product->data && product->rows > 0) {
        volatile double temp = product->data[0][0];
        if (product->rows > 1) temp += product->data[product->rows-1][product->cols-1];
        (void)temp;
    }
    
    long mem_after = get_memory_usage();
    PerformanceResult result = measure_performance(product, time_start, time_end, mem_before, mem_after, a->rows, name);
    
    matrix_free(product);
    return result;
}

// Multi-run benchmark for regular algorithms
PerformanceStats benchmark_algorithm_stats(Matrix* (*multiply_func)(Matrix*, Matrix*), 
                                          Matrix* a, Matrix* b, const char* name, int num_runs) {
    PerformanceResult* results = malloc(num_runs * sizeof(PerformanceResult));
    if (!results) return (PerformanceStats){0};
    
    printf("%s ", name);
    fflush(stdout);
    
    for (int i = 0; i < num_runs; i++) {
        results[i] = benchmark_algorithm(multiply_func, a, b, "");
        printf(".");
        fflush(stdout);
    }
    
    PerformanceStats stats = calculate_stats(results, num_runs);
    printf(" %.3fs ", stats.avg_time);
    
    free(results);
    return stats;
}

// Multi-run benchmark for cache algorithms
PerformanceStats benchmark_cache_algorithm_stats(Matrix* (*multiply_func)(Matrix*, Matrix*, int), 
                                                Matrix* a, Matrix* b, int block_size, const char* name, int num_runs) {
    PerformanceResult* results = malloc(num_runs * sizeof(PerformanceResult));
    if (!results) return (PerformanceStats){0};
    
    printf("%s ", name);
    fflush(stdout);
    
    for (int i = 0; i < num_runs; i++) {
        results[i] = benchmark_cache_algorithm(multiply_func, a, b, block_size, "");
        printf(".");
        fflush(stdout);
    }
    
    PerformanceStats stats = calculate_stats(results, num_runs);
    printf(" %.3fs ", stats.avg_time);
    
    free(results);
    return stats;
}

// Run comprehensive performance comparison
void run_performance_comparison(int max_size, int step_size __attribute__((unused))) {
    printf("\n=== PERFORMANCE COMPARISON ===\n");
    printf("Running benchmarks with detailed results exported to CSV files...\n");
    
    // Use specific matrix sizes: 256, 512, 1024, 2048
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        if (size > max_size) break; // Skip sizes larger than max_size
        
        // Create test matrices
        Matrix* a = matrix_create(size, size);
        Matrix* b = matrix_create(size, size);
        
        if (!a || !b) {
            printf("Failed to allocate matrices of size %d\n", size);
            matrix_free(a);
            matrix_free(b);
            continue;
        }
        
        matrix_fill_random(a, -1.0, 1.0);
        matrix_fill_random(b, -1.0, 1.0);
        
        // Test all algorithms with 3 runs each for statistics
        const int num_runs = 3;
        const int num_algorithms = 4;
        const char* algorithm_names[] = {"Basic", "Loop_Unroll", "Cache_Block", "Strassen"};
        
        // Allocate storage for all results
        PerformanceResult* all_results[num_algorithms];
        PerformanceStats stats[num_algorithms];
        
        for (int alg = 0; alg < num_algorithms; alg++) {
            all_results[alg] = malloc(num_runs * sizeof(PerformanceResult));
        }
        
        // Run all algorithms
        stats[0] = benchmark_algorithm_stats(matrix_multiply_basic, a, b, "Basic", num_runs);
        stats[1] = benchmark_algorithm_stats(matrix_multiply_loop_unroll, a, b, "Loop Unroll", num_runs);
        stats[2] = benchmark_cache_algorithm_stats(matrix_multiply_cache_block, a, b, 64, "Cache Block", num_runs);
        stats[3] = benchmark_algorithm_stats(matrix_multiply_strassen, a, b, "Strassen", num_runs);
        
        // Collect individual results for detailed export
        for (int r = 0; r < num_runs; r++) {
            all_results[0][r] = benchmark_algorithm(matrix_multiply_basic, a, b, "");
            all_results[1][r] = benchmark_algorithm(matrix_multiply_loop_unroll, a, b, "");
            all_results[2][r] = benchmark_cache_algorithm(matrix_multiply_cache_block, a, b, 64, "");
            all_results[3][r] = benchmark_algorithm(matrix_multiply_strassen, a, b, "");
        }
        
        // Export statistical summary to CSV
        char csv_filename[64];
        snprintf(csv_filename, sizeof(csv_filename), "dense_%dx%d.csv", size, size);
        export_dense_results_csv(csv_filename, size, stats, algorithm_names, num_algorithms);
        
        // Export detailed individual results to CSV
        export_detailed_dense_results_csv(csv_filename, size, all_results, algorithm_names, num_algorithms, num_runs);
        
        // Clean up allocated memory
        for (int alg = 0; alg < num_algorithms; alg++) {
            free(all_results[alg]);
        }
        
        // Find best algorithm
        double best_time = stats[0].avg_time;
        const char* best_algorithm = algorithm_names[0];
        for (int alg = 1; alg < num_algorithms; alg++) {
            if (stats[alg].avg_time < best_time) {
                best_time = stats[alg].avg_time;
                best_algorithm = algorithm_names[alg];
            }
        }
        
        printf("✓ %dx%d: Best=%s (%.3fs, %.1fx speedup)\n", 
               size, size, best_algorithm, best_time, 
               stats[0].avg_time / best_time);
        
        matrix_free(a);
        matrix_free(b);
    }
    
    // Create summary file
    create_summary_csv("benchmark_summary.csv");
    printf("\n✓ Dense matrix benchmarks complete. Summary→benchmark_summary.csv\n");
}