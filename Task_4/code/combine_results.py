#!/usr/bin/env python3
"""
Combine latest Python and Java benchmark results into a summary CSV.
Shows average, min, max, and standard deviation for each configuration.
"""

import pandas as pd
import numpy as np
import glob
import os

def get_latest_file(pattern):
    """Get the most recent file matching the pattern."""
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def calculate_stats(df, group_cols):
    """Calculate statistics for each group."""
    # First, convert data_transferred from KB to MB for the distributed method
    df_copy = df.copy()
    df_copy['data_transferred_mb'] = df_copy['data_transferred_kb'] / 1024.0
    
    stats = df_copy.groupby(group_cols).agg({
        'time_ms': ['mean', 'min', 'max', 'std'],
        'memory_kb': ['mean', 'min', 'max', 'std'],
        'network_time_ms': 'mean',
        'data_transferred_mb': 'mean',
        'cluster_nodes': 'first',
        'memory_per_node_kb': 'mean'
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stats.columns]
    
    return stats

def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all individual benchmark result files
    file_patterns = {
        'Java Basic': os.path.join(script_dir, 'java/results/benchmark_results_java_basic_*.csv'),
        'Java Parallel': os.path.join(script_dir, 'java/results/benchmark_results_java_parallel_*.csv'),
        'Java Distributed 2 nodes': os.path.join(script_dir, 'java/results/benchmark_results_java_distributed_2nodes_*.csv'),
        'Java Distributed 4 nodes': os.path.join(script_dir, 'java/results/benchmark_results_java_distributed_4nodes_*.csv'),
        'Python Basic': os.path.join(script_dir, 'python/results/benchmark_results_python_basic_*.csv'),
        'Python Parallel': os.path.join(script_dir, 'python/results/benchmark_results_python_parallel_*.csv'),
        'Python Distributed 2 nodes': os.path.join(script_dir, 'python/results/benchmark_results_python_distributed_2nodes_*.csv'),
        'Python Distributed 4 nodes': os.path.join(script_dir, 'python/results/benchmark_results_python_distributed_4nodes_*.csv'),
    }
    
    # Collect all dataframes
    all_dfs = []
    
    for name, pattern in file_patterns.items():
        file_path = get_latest_file(pattern)
        if file_path:
            print(f"Reading {name}: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            # Extract implementation from name
            df['implementation'] = name.split()[0]  # 'Java' or 'Python'
            all_dfs.append(df)
        else:
            print(f"Warning: No file found for {name}")
    
    if not all_dfs:
        print("Error: No results found")
        return
    
    # Combine all dataframes
    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Calculate statistics
    group_cols = ['implementation', 'matrix_size', 'method', 'cluster_nodes']
    all_stats = calculate_stats(df_all, group_cols)
    
    # Combine results (already combined above)
    combined = all_stats
    
    # Reorder columns for better readability
    column_order = [
        'implementation', 'matrix_size', 'method', 'cluster_nodes_first',
        'time_ms_mean', 'time_ms_min', 'time_ms_max', 'time_ms_std',
        'memory_kb_mean', 'memory_kb_min', 'memory_kb_max', 'memory_kb_std',
        'network_time_ms_mean', 'data_transferred_mb_mean',
        'memory_per_node_kb_mean'
    ]
    
    # Select only existing columns
    existing_cols = [col for col in column_order if col in combined.columns]
    combined = combined[existing_cols]
    
    # Rename columns for cleaner output
    combined = combined.rename(columns={
        'cluster_nodes_first': 'cluster_nodes',
        'memory_per_node_kb_mean': 'memory_per_node_kb'
    })
    
    # Round numeric columns
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].round(2)
    
    # Sort by implementation, method, cluster_nodes, matrix_size
    combined = combined.sort_values(['implementation', 'method', 'cluster_nodes', 'matrix_size'])
    
    # Save combined results
    output_dir = os.path.join(script_dir, 'combined_result')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'combined_benchmark_results_all.csv')
    combined.to_csv(output_file, index=False)
    
    print(f"\nCombined results saved to: {output_file}")
    print(f"Total rows: {len(combined)}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(combined.to_string(index=False))

if __name__ == '__main__':
    main()
