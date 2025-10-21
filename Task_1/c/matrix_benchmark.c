/*
 * C Matrix Multiplication Benchmark
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include "matrix_multiply.h"
#include "matrix_utils.h"
#include "stats_utils.h"
#include "csv_writer.h"

/**
 * Structure to hold benchmark results
 */
typedef struct {
    double* times;
    double* memory_usage;
    int num_runs;
} BenchmarkResult;

/**
 * Get current memory usage in KB
 */
long get_memory_usage_kb(void) {
    FILE* file = fopen("/proc/self/status", "r");
    if (file == NULL) return -1;
    
    char line[128];
    long vmrss = -1;
    
    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %ld kB", &vmrss);
            break;
        }
    }
    
    fclose(file);
    return vmrss;
}

/**
 * Benchmark matrix multiplication for a given size
 * size: Matrix size (n x n)
 * num_runs: Number of benchmark runs
 * Returns: BenchmarkResult with timing and memory data
 */
BenchmarkResult benchmark_matrix_multiply(int size, int num_runs) {
    BenchmarkResult result;
    result.times = (double*)malloc(num_runs * sizeof(double));
    result.memory_usage = (double*)malloc(num_runs * sizeof(double));
    result.num_runs = num_runs;
    
    struct timeval start, end;
    
    for (int run = 0; run < num_runs; run++) {
        // Get memory before
        long memory_before = get_memory_usage_kb();
        
        // Create matrices for this run
        double** A = create_random_matrix(size);
        double** B = create_random_matrix(size);
        
        // Measure execution time
        gettimeofday(&start, NULL);
        double** C = matrix_multiply(A, B, size);
        gettimeofday(&end, NULL);
        
        // Get memory after
        long memory_after = get_memory_usage_kb();
        double memory_used_mb = (memory_after - memory_before) / 1024.0;
        
        double execution_time = (end.tv_sec - start.tv_sec) + 
                               (end.tv_usec - start.tv_usec) / 1e6;
        
        result.times[run] = execution_time;
        result.memory_usage[run] = memory_used_mb > 0 ? memory_used_mb : 0;
        
        // Free memory
        free_matrix(A, size);
        free_matrix(B, size);
        free_matrix(C, size);
    }
    
    return result;
}

/**
 * Free benchmark result memory
 */
void free_benchmark_result(BenchmarkResult* result) {
    if (result->times) {
        free(result->times);
        result->times = NULL;
    }
    if (result->memory_usage) {
        free(result->memory_usage);
        result->memory_usage = NULL;
    }
}

/**
 * Run benchmark tests for different matrix sizes
 */
void run_benchmark(const char* csv_file) {
    // Test different matrix sizes
    int sizes[] = {64, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_runs = 5;
    
    printf("C Matrix Multiplication Benchmark\n");
    printf("==================================================\n");
    printf("Number of runs per size: %d\n", num_runs);
    printf("Matrix sizes: ");
    for (int i = 0; i < num_sizes; i++) {
        printf("%d", sizes[i]);
        if (i < num_sizes - 1) printf(", ");
    }
    printf("\n");
    if (csv_file != NULL) {
        printf("CSV output: %s\n", csv_file);
    }
    printf("\n");
    
    // Store all results for CSV output
    double** all_times = (double**)malloc(num_sizes * sizeof(double*));
    double** all_memory = (double**)malloc(num_sizes * sizeof(double*));
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("Testing matrix size %dx%d...\n", size, size);
        
        BenchmarkResult result = benchmark_matrix_multiply(size, num_runs);
        double* times = result.times;
        double* memory_usage = result.memory_usage;
        
        // Store for CSV output
        all_times[s] = (double*)malloc(num_runs * sizeof(double));
        all_memory[s] = (double*)malloc(num_runs * sizeof(double));
        memcpy(all_times[s], times, num_runs * sizeof(double));
        memcpy(all_memory[s], memory_usage, num_runs * sizeof(double));
        
        // Calculate statistics
        double avg_time = calculate_average(times, num_runs);
        double min_time = find_min(times, num_runs);
        double max_time = find_max(times, num_runs);
        double std_dev = calculate_std_dev(times, num_runs);
        
        double avg_memory = calculate_average(memory_usage, num_runs);
        double max_memory = find_max(memory_usage, num_runs);
        
        printf("  Times: [");
        for (int i = 0; i < num_runs; i++) {
            printf("%.4f", times[i]);
            if (i < num_runs - 1) printf(", ");
        }
        printf("] seconds\n");
        
        printf("  Average: %.4f seconds\n", avg_time);
        printf("  Min: %.4f seconds\n", min_time);
        printf("  Max: %.4f seconds\n", max_time);
        printf("  Std Dev: %.4f seconds\n", std_dev);
        
        printf("  Memory: [");
        for (int i = 0; i < num_runs; i++) {
            printf("%.1f", memory_usage[i]);
            if (i < num_runs - 1) printf(", ");
        }
        printf("] MB\n");
        
        printf("  Avg Memory: %.1f MB\n", avg_memory);
        printf("  Peak Memory: %.1f MB\n", max_memory);
        printf("\n");
        
        free_benchmark_result(&result);
    }
    
    // Write to CSV if specified
    if (csv_file != NULL) {
        write_benchmark_to_csv("C", sizes, num_sizes, all_times, all_memory, num_runs, csv_file);
    }
    
    // Free allocated memory
    for (int s = 0; s < num_sizes; s++) {
        free(all_times[s]);
        free(all_memory[s]);
    }
    free(all_times);
    free(all_memory);
}

int main(int argc, char* argv[]) {
    // Seed random number generator
    srand(42);
    
    // Check for CSV file argument
    const char* csv_file = (argc > 1) ? argv[1] : NULL;
    
    run_benchmark(csv_file);
    return 0;
}