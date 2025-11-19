#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Helper function to create results directory and open file
static FILE* open_results_file(const char* filename, const char* mode) {
    if (system("mkdir -p results") != 0) {
        printf("Warning: Could not create results directory\n");
    }
    char full_path[256];
    snprintf(full_path, sizeof(full_path), "results/%s", filename);
    
    FILE* file = fopen(full_path, mode);
    if (!file) {
        printf("Warning: Could not open CSV file %s\n", full_path);
    }
    return file;
}

// Export maximum size results to CSV
void export_max_size_results_csv(const char* filename, int max_basic, int max_loop, int max_cache, int max_strassen) {
    FILE* file = open_results_file(filename, "w");
    if (!file) return;
    
    // Get timestamp
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    fprintf(file, "Maximum_Matrix_Size_Analysis\n");
    fprintf(file, "Timestamp,%s\n", timestamp);
    fprintf(file, "Timeout_Seconds,1\n");
    fprintf(file, "Step_Size,10\n");
    fprintf(file, "Test_Range,800 to 5000\n");
    fprintf(file, "\n");
    fprintf(file, "Algorithm,Max_Matrix_Size,Max_Elements,Memory_Requirement_MB\n");
    
    const int sizes[] = {max_basic, max_loop, max_cache, max_strassen};
    const char* names[] = {"Basic", "Loop_Unroll", "Cache_Block", "Strassen"};
    
    for (int i = 0; i < 4; i++) {
        long elements = (long)sizes[i] * sizes[i];
        double memory_mb = (elements * sizeof(double) * 3) / (1024.0 * 1024.0); // 3 matrices: A, B, Result
        fprintf(file, "%s,%d,%ld,%.2f\n", names[i], sizes[i], elements, memory_mb);
    }
    
    fclose(file);
}

// CSV export functions for dense matrices
void export_dense_results_csv(const char* filename, int matrix_size, PerformanceStats* results, const char** algorithm_names, int num_algorithms) {
    FILE* file = open_results_file(filename, "w");
    if (!file) return;
    
    // Write header for statistics
    fprintf(file, "Matrix_Size,Algorithm,Avg_Time_s,StdDev_Time_s,Min_Time_s,Max_Time_s,Avg_Memory_KB,StdDev_Memory_KB,Min_Memory_KB,Max_Memory_KB,Speedup\n");
    
    // Write statistical data for each algorithm
    double baseline_time = results[0].avg_time; // Basic algorithm as baseline
    for (int i = 0; i < num_algorithms; i++) {
        double speedup = (baseline_time > 0) ? baseline_time / results[i].avg_time : 1.0;
        fprintf(file, "%d,%s,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%ld,%ld,%.2f\n",
                matrix_size, algorithm_names[i],
                results[i].avg_time, results[i].std_dev_time, results[i].min_time, results[i].max_time,
                results[i].avg_memory, results[i].std_dev_memory, results[i].min_memory, results[i].max_memory,
                speedup);
    }
    
    fclose(file);
}

void export_sparse_results_csv(const char* filename, int matrix_size, double sparsity, PerformanceStats dense, PerformanceStats sparse, int storage_dense, int storage_sparse, int is_first_entry) {
    FILE* file = open_results_file(filename, is_first_entry ? "w" : "a");
    if (!file) return;
    
    if (is_first_entry) {
        fprintf(file, "Matrix_Size,Sparsity_Percent,Format,Avg_Time_s,StdDev_Time_s,Min_Time_s,Max_Time_s,Avg_Memory_KB,StdDev_Memory_KB,Min_Memory_KB,Max_Memory_KB,Storage_Elements,Speedup,Runtime_Memory_Savings_Percent\n");
    }
    
    double speedup = (dense.avg_time > 0) ? dense.avg_time / sparse.avg_time : 1.0;
    // Calculate actual runtime memory savings (negative = CSR uses more memory)
    double memory_savings = ((double)(dense.avg_memory - sparse.avg_memory) / dense.avg_memory) * 100.0;
    
    // Write dense and sparse results
    const PerformanceStats stats[] = {dense, sparse};
    const char* formats[] = {"Dense", "Sparse"};
    const int storages[] = {storage_dense, storage_sparse};
    const double speedups[] = {1.00, speedup};
    const double savings[] = {0.0, memory_savings};
    
    for (int i = 0; i < 2; i++) {
        fprintf(file, "%d,%.0f,%s,%.6f,%.6f,%.6f,%.6f,%ld,%ld,%ld,%ld,%d,%.2f,%.1f\n",
                matrix_size, sparsity, formats[i],
                stats[i].avg_time, stats[i].std_dev_time, stats[i].min_time, stats[i].max_time,
                stats[i].avg_memory, stats[i].std_dev_memory, stats[i].min_memory, stats[i].max_memory, 
                storages[i], speedups[i], savings[i]);
    }
    
    fclose(file);
}

void export_detailed_sparse_results_csv(const char* filename, int matrix_size, double sparsity, PerformanceResult* dense_results, PerformanceResult* sparse_results, int num_runs, int is_first_entry) {
    char detailed_filename[256];
    snprintf(detailed_filename, sizeof(detailed_filename), "detailed_%s", filename);
    
    FILE* file = open_results_file(detailed_filename, is_first_entry ? "w" : "a");
    if (!file) return;
    
    if (is_first_entry) {
        fprintf(file, "Matrix_Size,Sparsity_Percent,Format,Run_Number,Execution_Time_s,Memory_Usage_KB\n");
    }
    
    // Write individual run data for both dense and sparse
    const PerformanceResult* results[] = {dense_results, sparse_results};
    const char* formats[] = {"Dense", "Sparse"};
    
    for (int format = 0; format < 2; format++) {
        for (int run = 0; run < num_runs; run++) {
            fprintf(file, "%d,%.0f,%s,%d,%.6f,%ld\n",
                    matrix_size, sparsity, formats[format], run + 1,
                    results[format][run].execution_time, results[format][run].memory_usage);
        }
    }
    
    fclose(file);
}

void export_detailed_dense_results_csv(const char* filename, int matrix_size, PerformanceResult** all_results, const char** algorithm_names, int num_algorithms, int num_runs) {
    char detailed_filename[256];
    snprintf(detailed_filename, sizeof(detailed_filename), "detailed_%s", filename);
    
    FILE* file = open_results_file(detailed_filename, "w");
    if (!file) return;
    
    // Write header and individual run data
    fprintf(file, "Matrix_Size,Algorithm,Run_Number,Execution_Time_s,Memory_Usage_KB\n");
    
    for (int i = 0; i < num_algorithms; i++) {
        for (int j = 0; j < num_runs; j++) {
            fprintf(file, "%d,%s,%d,%.6f,%ld\n",
                    matrix_size, algorithm_names[i], j + 1,
                    all_results[i][j].execution_time, all_results[i][j].memory_usage);
        }
    }
    
    fclose(file);
}

void create_summary_csv(const char* filename) {
    FILE* file = open_results_file(filename, "w");
    if (!file) return;
    
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    const char* summary_data[] = {
        "Benchmark_Summary",
        "Timestamp,%s",
        "Matrix_Sizes,256x256 512x512 1024x1024 2048x2048", 
        "Algorithms,Basic Loop_Unroll Cache_Block Strassen",
        "Runs_Per_Algorithm,3",
        "Sparse_Analysis,50%% 70%% 90%% 95%% sparsity levels",
        "Memory_Measurement,Real RSS from /proc/self/status",
        "Compiler_Flags,-O3 -march=native",
        "",
        "Files_Generated:",
        "dense_256x256.csv,Dense matrix results for 256x256", 
        "dense_512x512.csv,Dense matrix results for 512x512",
        "dense_1024x1024.csv,Dense matrix results for 1024x1024",
        "dense_2048x2048.csv,Dense matrix results for 2048x2048",
        "sparse_results.csv,Sparse matrix analysis all sparsity levels",
        "detailed_sparse_results.csv,Individual run data for sparse analysis"
    };
    
    for (size_t i = 0; i < sizeof(summary_data) / sizeof(summary_data[0]); i++) {
        if (i == 1) {
            fprintf(file, summary_data[i], timestamp);
        } else {
            fprintf(file, "%s", summary_data[i]);  
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}