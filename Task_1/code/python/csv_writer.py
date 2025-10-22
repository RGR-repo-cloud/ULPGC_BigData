#!/usr/bin/env python3
"""
CSV Results Writer for Matrix Multiplication Benchmarks
"""
import csv
import sys


def write_benchmark_to_csv(language, results, csv_file):
    """
    Write benchmark results to CSV file
    """
    
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        
        for size, data in results.items():
            times = data['times']
            memory_peaks = data['memory_peaks']
            
            # Write each run as a separate row
            for run_num, (time_val, memory_val) in enumerate(zip(times, memory_peaks), 1):
                writer.writerow([
                    language,
                    size,
                    run_num,
                    f"{time_val:.6f}",
                    f"{memory_val:.2f}"
                ])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 csv_writer.py <language> <results_dict> <csv_file>")
        sys.exit(1)
    
    language = sys.argv[1]
    print(f"CSV writer ready for {language}")