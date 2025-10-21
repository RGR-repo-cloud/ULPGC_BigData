"""
Python Matrix Multiplication Benchmark
TESTING CODE ONLY - Performance testing and measurement
"""
import time
import tracemalloc
import psutil
import os
import csv
import sys
from matrix_multiply import matrix_multiply
from matrix_utils import create_random_matrix
from stats_utils import calculate_statistics
from csv_writer import write_benchmark_to_csv


def benchmark_matrix_multiply(size, num_runs=5):
    """
    Benchmark matrix multiplication for a given size
    Returns: dict with timing and memory statistics
    """
    times = []
    memory_peaks = []
    
    for run in range(num_runs):
        # Create matrices for this run
        A = create_random_matrix(size)
        B = create_random_matrix(size)
        
        # Start memory and time measurement
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        matrix_multiply(A, B)
        end_time = time.time()
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        tracemalloc.stop()
        
        execution_time = end_time - start_time
        memory_used = max(peak / 1024 / 1024, memory_after - memory_before)  # MB
        
        times.append(execution_time)
        memory_peaks.append(memory_used)
    
    return {
        'times': times,
        'memory_peaks': memory_peaks
    }


def write_csv_results(results, csv_file=None):
    """Write results to CSV file using the csv_writer module"""
    if csv_file is None:
        return
    
    try:
        write_benchmark_to_csv('Python', results, csv_file)
    except Exception as e:
        print(f"Warning: Could not write to CSV file: {e}")


def run_benchmark():
    """Run benchmark tests for different matrix sizes"""

    sizes = [64, 128, 256, 512, 1024]  
    num_runs = 5
    
    # Check if CSV file is provided as command line argument
    csv_file = None
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    print("Python Matrix Multiplication Benchmark")
    print("="*50)
    print(f"Number of runs per size: {num_runs}")
    print(f"Matrix sizes: {sizes}")
    if csv_file:
        print(f"CSV output: {csv_file}")
    print()
    
    results = {}
    
    for size in sizes:
        print(f"Testing matrix size {size}x{size}...")
        
        result = benchmark_matrix_multiply(size, num_runs)
        times = result['times']
        memory_peaks = result['memory_peaks']
        
        # Calculate statistics using utility functions
        time_stats = calculate_statistics(times)
        memory_stats = calculate_statistics(memory_peaks)
        
        results[size] = {
            'times': times,
            'avg': time_stats['avg'],
            'min': time_stats['min'],
            'max': time_stats['max'],
            'std': time_stats['std_dev'],
            'memory_peaks': memory_peaks,
            'avg_memory': memory_stats['avg'],
            'max_memory': memory_stats['max']
        }
        
        print(f"  Times: {[f'{t:.4f}' for t in times]} seconds")
        print(f"  Average: {time_stats['avg']:.4f} seconds")
        print(f"  Min: {time_stats['min']:.4f} seconds")
        print(f"  Max: {time_stats['max']:.4f} seconds")
        print(f"  Std Dev: {time_stats['std_dev']:.4f} seconds")
        print(f"  Memory: {[f'{m:.1f}' for m in memory_peaks]} MB")
        print(f"  Avg Memory: {memory_stats['avg']:.1f} MB")
        print(f"  Peak Memory: {memory_stats['max']:.1f} MB")
        print()
    
    # Write results to CSV if specified
    write_csv_results(results, csv_file)
    
    return results


if __name__ == "__main__":
    run_benchmark()