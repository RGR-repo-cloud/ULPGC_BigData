/*
 * CSV Writer for C Matrix Multiplication Benchmark - Header
 */
#ifndef CSV_WRITER_H
#define CSV_WRITER_H

void write_benchmark_to_csv(const char* language, int* sizes, int num_sizes, 
                           double** all_times, double** all_memory, 
                           int num_runs, const char* csv_file);

#endif