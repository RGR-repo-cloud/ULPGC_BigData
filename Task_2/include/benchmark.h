#ifndef BENCHMARK_H
#define BENCHMARK_H

#define _POSIX_C_SOURCE 199309L
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>
#include "matrix.h"
#include "sparse_matrix.h"

// Performance measurement structure
typedef struct {
    double execution_time;
    long memory_usage;      // Peak memory usage in KB
    int max_matrix_size;    // Maximum size tested successfully
} PerformanceResult;

// Statistics structure for multiple runs
typedef struct {
    double avg_time;
    double std_dev_time;
    double min_time;
    double max_time;
    long avg_memory;
    long std_dev_memory;
    long min_memory;
    long max_memory;
    int num_runs;
} PerformanceStats;

// Benchmarking functions
PerformanceResult benchmark_algorithm(Matrix* (*multiply_func)(Matrix*, Matrix*), 
                                    Matrix* a, Matrix* b, const char* name);
PerformanceResult benchmark_cache_algorithm(Matrix* (*multiply_func)(Matrix*, Matrix*, int), 
                                          Matrix* a, Matrix* b, int block_size, const char* name);
PerformanceResult benchmark_sparse_algorithm(CSRMatrix* a, CSRMatrix* b, const char* name);

// Multi-run statistical benchmarking functions
PerformanceStats benchmark_algorithm_stats(Matrix* (*multiply_func)(Matrix*, Matrix*), 
                                          Matrix* a, Matrix* b, const char* name, int num_runs);
PerformanceStats benchmark_cache_algorithm_stats(Matrix* (*multiply_func)(Matrix*, Matrix*, int), 
                                                Matrix* a, Matrix* b, int block_size, const char* name, int num_runs);
PerformanceStats benchmark_sparse_algorithm_stats(CSRMatrix* a, CSRMatrix* b, const char* name, int num_runs);

// Statistical calculation functions
PerformanceStats calculate_stats(PerformanceResult* results, int num_runs);

// CSV export functions
void export_dense_results_csv(const char* filename, int matrix_size, PerformanceStats* stats, const char** algorithm_names, int num_algorithms);
void export_detailed_dense_results_csv(const char* filename, int matrix_size, PerformanceResult** all_results, const char** algorithm_names, int num_algorithms, int num_runs);
void export_sparse_results_csv(const char* filename, int matrix_size, double sparsity, PerformanceStats dense, PerformanceStats sparse, int storage_dense, int storage_sparse, int is_first_entry);
void export_detailed_sparse_results_csv(const char* filename, int matrix_size, double sparsity, PerformanceResult* dense_results, PerformanceResult* sparse_results, int num_runs, int is_first_entry);
void create_summary_csv(const char* filename);
void run_performance_comparison(int max_size, int step_size);
void run_sparsity_analysis(int matrix_size, double min_sparsity, double max_sparsity, int steps);
void print_performance_header(void);
void print_performance_result(const char* algorithm, int size, PerformanceResult result);

// (Real-world sparse benchmark removed)

// Utility functions
double get_wall_time(void);
long get_memory_usage(void);
PerformanceResult measure_performance(void* result, double time_start, double time_end, long mem_before, long mem_after, int matrix_size, const char* name);
int find_max_matrix_size(Matrix* (*multiply_func)(Matrix*, Matrix*));
int find_max_matrix_size_cache(Matrix* (*multiply_func)(Matrix*, Matrix*, int), int block_size);
void test_maximum_matrix_sizes(void);
void export_max_size_results_csv(const char* filename, int max_basic, int max_loop, int max_cache, int max_strassen);

#endif // BENCHMARK_H