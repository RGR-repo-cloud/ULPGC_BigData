/*
 * CSV Writer for C Matrix Multiplication Benchmark
 */
#include <stdio.h>
#include <string.h>
#include "csv_writer.h"

/**
 * Write benchmark results to CSV file
 */
void write_benchmark_to_csv(const char* language, int* sizes, int num_sizes, 
                           double** all_times, double** all_memory, 
                           int num_runs, const char* csv_file) {
    
    if (csv_file == NULL) return;
    
    FILE* file = fopen(csv_file, "a");
    if (file == NULL) {
        fprintf(stderr, "Warning: Could not open CSV file for writing\n");
        return;
    }
    
    // Write data for each size and run
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        double* times = all_times[s];
        double* memory = all_memory[s];
        
        for (int run = 0; run < num_runs; run++) {
            fprintf(file, "%s,%d,%d,%.6f,%.2f\n",
                    language, size, run + 1, times[run], memory[run]);
        }
    }
    
    fclose(file);
}