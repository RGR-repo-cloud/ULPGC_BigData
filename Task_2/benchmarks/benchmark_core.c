#define _POSIX_C_SOURCE 199309L
#include "benchmark.h"
#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <float.h>

// Get current wall time in seconds
double get_wall_time(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec + ts.tv_nsec / 1e9;
    }
    return 0.0;
}

// Get current memory usage in KB from /proc/self/status
long get_memory_usage(void) {
    FILE* file = fopen("/proc/self/status", "r");
    if (!file) return 0;
    
    char line[256];
    long rss_kb = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %ld kB", &rss_kb);
            break;
        }
    }
    
    fclose(file);
    return rss_kb;
}

// Common timing and memory measurement wrapper
PerformanceResult measure_performance(void* result, double time_start, double time_end, long mem_before, long mem_after, int matrix_size, const char* name) {
    PerformanceResult perf_result = {0.0, 0, 0};
    
    if (result) {
        perf_result.execution_time = time_end - time_start;
        perf_result.memory_usage = mem_after - mem_before;
        perf_result.max_matrix_size = matrix_size;
        if (strlen(name) > 0) {
            printf("%.3f s (mem: %ld KB)\n", perf_result.execution_time, perf_result.memory_usage);
        }
    } else {
        if (strlen(name) > 0) {
            printf("Failed or timeout\n");
        }
        perf_result.execution_time = -1.0;
        perf_result.memory_usage = 0;
        perf_result.max_matrix_size = 0;
    }
    
    return perf_result;
}

void print_performance_header(void) {
    printf("%-15s %-8s %-12s %-12s %-15s\n", "Algorithm", "Size", "Time (s)", "Memory (KB)", "GFLOPS");
    printf("%-15s %-8s %-12s %-12s %-15s\n", "=========", "====", "========", "===========", "======");
}

void print_performance_result(const char* algorithm, int size, PerformanceResult result) {
    double gflops = (2.0 * size * size * size) / (result.execution_time * 1e9);
    printf("%-15s %-8d %-12.3f %-12ld %-15.2f\n", algorithm, size, result.execution_time, result.memory_usage, gflops);
}

PerformanceStats calculate_stats(PerformanceResult* results, int num_runs) {
    PerformanceStats stats = {0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0};
    
    if (num_runs == 0) return stats;
    
    // Calculate mean
    double sum_time = 0.0;
    long sum_memory = 0;
    
    for (int i = 0; i < num_runs; i++) {
        sum_time += results[i].execution_time;
        sum_memory += results[i].memory_usage;
    }
    
    stats.avg_time = sum_time / num_runs;
    stats.avg_memory = (double)sum_memory / num_runs;
    
    // Find min and max
    stats.min_time = results[0].execution_time;
    stats.max_time = results[0].execution_time;
    
    for (int i = 1; i < num_runs; i++) {
        if (results[i].execution_time < stats.min_time) {
            stats.min_time = results[i].execution_time;
        }
        if (results[i].execution_time > stats.max_time) {
            stats.max_time = results[i].execution_time;
        }
    }
    
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    for (int i = 0; i < num_runs; i++) {
        double diff = results[i].execution_time - stats.avg_time;
        sum_sq_diff += diff * diff;
    }
    
    stats.std_dev_time = sqrt(sum_sq_diff / num_runs);
    stats.num_runs = num_runs;
    
    return stats;
}