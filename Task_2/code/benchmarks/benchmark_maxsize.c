#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// Find maximum matrix size that can be handled efficiently
int find_max_matrix_size(Matrix* (*multiply_func)(Matrix*, Matrix*)) {
    int size = 800;
    int max_successful_size = 0;
    const double timeout = 1.0; // 1 second timeout for efficiency
    double last_elapsed = 0.0;
    int hit_timeout = 0;
    
    printf("Finding maximum matrix size for algorithm...\n");
    
    while (size <= 5000) { // Reasonable upper limit
        Matrix* a = matrix_create(size, size);
        Matrix* b = matrix_create(size, size);
        
        if (!a || !b) {
            matrix_free(a);
            matrix_free(b);
            break;
        }
        
        matrix_fill_random(a, -1.0, 1.0);
        matrix_fill_random(b, -1.0, 1.0);
        
        // Run multiple times and take minimum to reduce noise
        double min_elapsed = INFINITY;
        const int num_warmup = 1;
        const int num_runs = 3;
        
        // Warmup run to stabilize caches
        for (int w = 0; w < num_warmup; w++) {
            Matrix* warmup_result = multiply_func(a, b);
            matrix_free(warmup_result);
        }
        
        // Actual timing runs
        Matrix* result = NULL;
        for (int r = 0; r < num_runs; r++) {
            double time_start = get_wall_time();
            result = multiply_func(a, b);
            double time_end = get_wall_time();
            
            double elapsed = time_end - time_start;
            if (elapsed < min_elapsed) {
                min_elapsed = elapsed;
            }
            
            if (r < num_runs - 1) { // Don't free the last result
                matrix_free(result);
            }
        }
        
        double elapsed = min_elapsed;
        if (result && elapsed <= timeout) {
            max_successful_size = size;
            last_elapsed = elapsed;
            matrix_free(result);
            printf("  Size %d: OK (%.3f seconds)\n", size, elapsed);
        } else {
            printf("  Size %d: Failed or timeout (%.3f seconds)\n", size, elapsed);
            hit_timeout = 1;
            matrix_free(result);
            matrix_free(a);
            matrix_free(b);
            break;
        }
        
        matrix_free(a);
        matrix_free(b);
        
        // Adaptive step size based on cache hierarchy
        if (size < 512) {
            size += 16; // Fine-grained steps in L3 cache range
        } else if (size < 1024) {
            size += 32; // Medium steps beyond L3
        } else {
            size += 64; // Larger steps in main memory range
        }
    }
    if (!hit_timeout && max_successful_size > 0) {
        printf("(Upper limit reached: did not hit 1s timeout)\n");
    }
    printf("Max efficient size: %d (%.3f seconds)\n", max_successful_size, last_elapsed);
    return max_successful_size;
}

// Find maximum matrix size for cache-blocked algorithms
int find_max_matrix_size_cache(Matrix* (*multiply_func)(Matrix*, Matrix*, int), int block_size) {
    int size = 800;
    int max_successful_size = 0;
    const double timeout = 1.0; // 1 second timeout for efficiency
    double last_elapsed = 0.0;
    int hit_timeout = 0;
    
    printf("Finding maximum matrix size for cache-blocked algorithm (block_size=%d)...\n", block_size);
    
    while (size <= 5000) {
        Matrix* a = matrix_create(size, size);
        Matrix* b = matrix_create(size, size);
        
        if (!a || !b) {
            matrix_free(a);
            matrix_free(b);
            break;
        }
        
        matrix_fill_random(a, -1.0, 1.0);
        matrix_fill_random(b, -1.0, 1.0);
        
        // Run multiple times and take minimum to reduce noise
        double min_elapsed = INFINITY;
        const int num_warmup = 1;
        const int num_runs = 3;
        
        // Warmup run to stabilize caches
        for (int w = 0; w < num_warmup; w++) {
            Matrix* warmup_result = multiply_func(a, b, block_size);
            matrix_free(warmup_result);
        }
        
        // Actual timing runs
        Matrix* result = NULL;
        for (int r = 0; r < num_runs; r++) {
            double time_start = get_wall_time();
            result = multiply_func(a, b, block_size);
            double time_end = get_wall_time();
            
            double elapsed = time_end - time_start;
            if (elapsed < min_elapsed) {
                min_elapsed = elapsed;
            }
            
            if (r < num_runs - 1) { // Don't free the last result
                matrix_free(result);
            }
        }
        
        double elapsed = min_elapsed;
        if (result && elapsed <= timeout) {
            max_successful_size = size;
            last_elapsed = elapsed;
            matrix_free(result);
            printf("  Size %d: OK (%.3f seconds)\n", size, elapsed);
        } else {
            printf("  Size %d: Failed or timeout (%.3f seconds)\n", size, elapsed);
            hit_timeout = 1;
            matrix_free(result);
            matrix_free(a);
            matrix_free(b);
            break;
        }
        
        matrix_free(a);
        matrix_free(b);
        
        // Adaptive step size based on cache hierarchy
        if (size < 512) {
            size += 16; // Fine-grained steps in L3 cache range
        } else if (size < 1024) {
            size += 32; // Medium steps beyond L3
        } else {
            size += 64; // Larger steps in main memory range
        }
    }
    if (!hit_timeout && max_successful_size > 0) {
        printf("(Upper limit reached: did not hit 1s timeout)\n");
    }
    printf("Max efficient size: %d (%.3f seconds)\n", max_successful_size, last_elapsed);
    return max_successful_size;
}

// Test maximum matrix sizes for all algorithms
void test_maximum_matrix_sizes(void) {
    printf("\n=== MAXIMUM MATRIX SIZE ANALYSIS ===\n");
    printf("Testing maximum efficiently handleable matrix sizes (1s timeout per size, adaptive steps from 800)...\n\n");
    
    // Test each algorithm
    printf("🔹 BASIC ALGORITHM:\n");
    int max_basic = find_max_matrix_size(matrix_multiply_basic);
    
    printf("\n🔹 LOOP UNROLL ALGORITHM:\n");
    int max_loop = find_max_matrix_size(matrix_multiply_loop_unroll);
    
    printf("\n🔹 CACHE BLOCK ALGORITHM:\n");
    int max_cache = find_max_matrix_size_cache(matrix_multiply_cache_block, 64);
    
    printf("\n🔹 STRASSEN ALGORITHM:\n");
    int max_strassen = find_max_matrix_size(matrix_multiply_strassen);
    
    // Print summary
    printf("\n=== MAXIMUM SIZE SUMMARY ===\n");
    printf("%-20s %10s %15s\n", "Algorithm", "Max Size", "Max Elements");
    printf("%-20s %10s %15s\n", "---------", "--------", "------------");
    printf("%-20s %10d %15ld\n", "Basic", max_basic, (long)max_basic * max_basic);
    printf("%-20s %10d %15ld\n", "Loop Unroll", max_loop, (long)max_loop * max_loop);
    printf("%-20s %10d %15ld\n", "Cache Block", max_cache, (long)max_cache * max_cache);
    printf("%-20s %10d %15ld\n", "Strassen", max_strassen, (long)max_strassen * max_strassen);
    
    // Export to CSV
    export_max_size_results_csv("max_matrix_sizes.csv", max_basic, max_loop, max_cache, max_strassen);
    
    printf("\n✓ Maximum size analysis complete. Results→max_matrix_sizes.csv\n");
}