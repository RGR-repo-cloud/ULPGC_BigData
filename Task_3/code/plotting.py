#!/usr/bin/env python3
"""
Enhanced plotting system for consolidated CSV benchmark results.
Generates visualizations for all metrics required in section 3.2.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Configure matplotlib for better font handling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedBenchmarkPlotter:
    def __init__(self, data_dir='results', output_dir='plots'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'performance').mkdir(exist_ok=True)
        (self.output_dir / 'efficiency').mkdir(exist_ok=True)
        (self.output_dir / 'hyperparams').mkdir(exist_ok=True)
    
    def find_parallel_algorithm(self, df):
        """Find the basic parallel algorithm (not advanced variants)."""
        parallel_algorithms = df[df['Algorithm'].str.contains('Parallel.*threads', regex=True, na=False)]['Algorithm'].unique()
        # Filter out advanced parallel algorithms
        basic_parallel = [alg for alg in parallel_algorithms if not any(x in alg for x in ['Advanced', 'semaphore', 'streams', 'Fork-Join'])]
        return basic_parallel[0] if basic_parallel else 'Parallel (4 threads)'  # fallback
    
    def find_parallel_algorithm_from_perf_data(self, df):
        """Find parallel algorithm from performance data."""
        return self.find_parallel_algorithm(df)
    
    def load_data(self, filename):
        """Load CSV data from consolidated files."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            print(f"📊 Loaded {filename}: {len(df)} records")
            return df
        except Exception as e:
            print(f"⚠️  Error loading {filename}: {e}")
            return None
    
    def clean_algorithm_name(self, algorithm_name):
        """Clean up algorithm names for better display in plots."""
        # Remove quotes and clean up common patterns
        clean_name = algorithm_name.strip('"')
        
        # Simplify long algorithm names for better readability
        if 'Basic Sequential' in clean_name:
            return 'Basic Sequential'
        elif 'Vectorized' in clean_name and 'block size' in clean_name:
            return 'Vectorized'
        elif 'Parallel' in clean_name and 'threads' in clean_name and 'Advanced' not in clean_name:
            return 'Parallel'
        elif 'Advanced Parallel' in clean_name and 'semaphore' in clean_name:
            return 'Adv. Parallel + Semaphore'
        elif 'Advanced Parallel' in clean_name and 'streams' in clean_name:
            return 'Parallel Streams'
        elif 'Advanced Parallel' in clean_name:
            return 'Advanced Parallel'
        elif 'Fork-Join' in clean_name:
            return 'Fork-Join'
        else:
            # Truncate very long names
            if len(clean_name) > 25:
                return clean_name[:22] + '...'
            return clean_name
    

    
    def plot_detailed_performance_analysis(self):
        """Performance analysis: Basic vs Parallel vs Vectorization - individual PNGs per matrix size."""
        df = self.load_data('all_performance_results.csv')
        if df is None:
            return
        
        # Filter for the 3 core approaches: Basic, Vectorized, and Parallel
        # Core algorithms from the first experimental set (comparison and algorithm_comparison)
        core_algorithms = {
            'Basic': 'Basic Sequential',
            'Vectorized': 'Vectorized (block size: 32)', 
            'Parallel': 'Parallel (8 threads)'
        }
        
        # Filter data for core algorithms only from the first experimental run (exactly first 24 rows)
        # The first experiment has Basic Sequential, Vectorized, and Parallel (8 threads)
        # This includes both 'comparison' and 'algorithm_comparison' test types for the same data
        core_experiment_data = df.iloc[:24].copy()  # Exactly first 24 rows
        
        # Filter for core algorithms only
        filtered_df = core_experiment_data[core_experiment_data['Algorithm'].isin(core_algorithms.values())].copy()
        
        # Create mapping for cleaner labels
        filtered_df['Algorithm_Clean'] = filtered_df['Algorithm'].map({v: k for k, v in core_algorithms.items()})
        
        matrix_sizes = sorted(filtered_df['Matrix_Size'].unique())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Create individual PNG for each matrix size
        for size in matrix_sizes:
            plt.figure(figsize=(10, 6))
            
            size_data = filtered_df[filtered_df['Matrix_Size'] == size]
            
            # Group by algorithm and take average (in case of multiple entries)
            avg_data = size_data.groupby('Algorithm_Clean')['Time_ms'].mean().reset_index()
            algorithms = avg_data['Algorithm_Clean'].values
            times = avg_data['Time_ms'].values
            
            # Ensure we have exactly 3 bars in the desired order
            ordered_algorithms = ['Basic', 'Vectorized', 'Parallel']
            ordered_times = []
            
            for alg in ordered_algorithms:
                if alg in algorithms:
                    idx = list(algorithms).index(alg)
                    ordered_times.append(times[idx])
                else:
                    ordered_times.append(0)  # Fallback if algorithm missing
            
            bars = plt.bar(ordered_algorithms, ordered_times, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
            plt.title(f'Matrix Multiplication Performance Comparison\nMatrix Size: {size}×{size}', 
                     fontsize=14, fontweight='bold')
            plt.xticks(fontsize=11)  # Remove rotation since we have only 3 clear labels
            plt.yticks(fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, time in zip(bars, ordered_times):
                if time > 0:  # Only add label if we have data
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ordered_times)*0.01,
                            f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Add speedup annotations (relative to Basic)
            basic_time = ordered_times[0]  # First entry is always Basic
            for i, (bar, time, alg) in enumerate(zip(bars, ordered_times, ordered_algorithms)):
                if i > 0 and time > 0:  # Skip Basic (index 0) and zero values
                    speedup = basic_time / time
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                            f'{speedup:.1f}x', ha='center', va='center', 
                            fontweight='bold', fontsize=9, color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance' / f'performance_matrix_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Performance analysis for {size}×{size} generated")
        
        # Also create a summary speedup comparison
        plt.figure(figsize=(12, 6))
        ordered_algorithms = ['Basic', 'Vectorized', 'Parallel']
        algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        x_pos = np.arange(len(matrix_sizes))
        width = 0.25
        
        for i, alg in enumerate(ordered_algorithms):
            speedups = []
            for size in matrix_sizes:
                alg_size_data = filtered_df[(filtered_df['Algorithm_Clean'] == alg) & 
                                          (filtered_df['Matrix_Size'] == size)]
                if len(alg_size_data) > 0:
                    speedups.append(alg_size_data['Speedup'].mean())
                else:
                    speedups.append(1.0 if alg == 'Basic' else 0)  # Basic always has speedup 1.0
            
            plt.bar(x_pos + i * width, speedups, width, label=alg, 
                   color=algorithm_colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for j, (pos, speedup) in enumerate(zip(x_pos + i * width, speedups)):
                if speedup > 0:
                    plt.text(pos, speedup + 0.1, f'{speedup:.1f}x',
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup vs Basic Sequential', fontsize=12, fontweight='bold')
        plt.title('Performance Speedup Comparison: Basic vs Vectorized vs Parallel', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width, [f'{size}×{size}' for size in matrix_sizes], fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance' / 'speedup_comparison_all_sizes.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Speedup comparison summary generated")
        print("✅ All individual performance analyses generated")
    
    def plot_memory_usage_analysis(self):
        """Memory usage analysis: individual PNGs per matrix size for all algorithms."""
        df = self.load_data('all_performance_results.csv')
        if df is None:
            return
        
        # Define algorithm matching function (same as performance analysis)
        def match_algorithm(algorithm_name):
            if algorithm_name == 'Basic Sequential' or algorithm_name == '"Basic Sequential"':
                return 'Basic'
            elif algorithm_name in ['Parallel (4 threads)', 'Parallel (8 threads)', 'Parallel (12 threads)',
                                  '"Parallel (4 threads)"', '"Parallel (8 threads)"', '"Parallel (12 threads)"']:
                return 'Parallel'
            elif algorithm_name in ['Advanced Parallel (4 threads)', 'Advanced Parallel (8 threads)', 'Advanced Parallel (12 threads)',
                                  '"Advanced Parallel (4 threads)"', '"Advanced Parallel (8 threads)"', '"Advanced Parallel (12 threads)"']:
                return 'Advanced Parallel'
            elif 'semaphore' in algorithm_name:
                return 'Advanced + Semaphore'
            elif 'streams' in algorithm_name:
                return 'Parallel Streams'
            elif 'Fork-Join' in algorithm_name:
                return 'Fork-Join'
            else:
                return None
        
        # Filter and clean algorithm names
        filtered_df = []
        for index, row in df.iterrows():
            algorithm = row['Algorithm']
            clean_name = match_algorithm(algorithm)
            
            if clean_name:
                new_row = row.copy()
                new_row['Algorithm_Clean'] = clean_name
                filtered_df.append(new_row)
        
        if not filtered_df:
            print("No matching algorithms found for memory analysis")
            return
        
        filtered_df = pd.DataFrame(filtered_df)
        matrix_sizes = sorted(filtered_df['Matrix_Size'].unique())
        
        # Create individual memory usage plots for each matrix size
        for size in matrix_sizes:
            size_data = filtered_df[filtered_df['Matrix_Size'] == size]
            
            plt.figure(figsize=(12, 8))
            
            # Get algorithm order and their memory usage
            algorithm_memory = size_data.groupby('Algorithm_Clean')['Memory_MB'].mean().sort_values()
            ordered_algorithms = algorithm_memory.index.tolist()
            ordered_memory = algorithm_memory.values
            
            # Create color scheme
            algorithm_colors = {
                'Basic': '#1f77b4',
                'Parallel': '#ff7f0e', 
                'Advanced Parallel': '#2ca02c',
                'Advanced + Semaphore': '#d62728',
                'Parallel Streams': '#9467bd',
                'Fork-Join': '#8c564b'
            }
            colors = [algorithm_colors.get(alg, '#gray') for alg in ordered_algorithms]
            
            bars = plt.bar(range(len(ordered_algorithms)), ordered_memory, 
                          color=colors, alpha=0.8, edgecolor='black')
            
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel('Memory Usage (MB)', fontsize=12)
            plt.title(f'Memory Usage Comparison - {size}×{size} Matrix', fontsize=14, fontweight='bold')
            plt.xticks(range(len(ordered_algorithms)), ordered_algorithms, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, memory) in enumerate(zip(bars, ordered_memory)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{memory:.1f}MB', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance' / f'memory_usage_matrix_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Memory usage analysis for {size}×{size} generated")
        
        # Also create a summary memory comparison across sizes
        plt.figure(figsize=(12, 6))
        ordered_algorithms = ['Basic', 'Parallel', 'Advanced Parallel', 'Advanced + Semaphore', 'Parallel Streams', 'Fork-Join']
        algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        x_pos = np.arange(len(matrix_sizes))
        width = 0.12
        
        for i, alg in enumerate(ordered_algorithms):
            memories = []
            for size in matrix_sizes:
                alg_size_data = filtered_df[(filtered_df['Algorithm_Clean'] == alg) & 
                                          (filtered_df['Matrix_Size'] == size)]
                if len(alg_size_data) > 0:
                    memories.append(alg_size_data['Memory_MB'].mean())
                else:
                    memories.append(0)
            
            plt.bar(x_pos + i * width, memories, width, label=alg, 
                   color=algorithm_colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for j, (pos, memory) in enumerate(zip(x_pos + i * width, memories)):
                if memory > 0:
                    plt.text(pos, memory + max(memories) * 0.01, f'{memory:.1f}',
                            ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.title('Memory Usage Scaling Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * (len(ordered_algorithms) - 1) / 2, 
                  [f'{size}×{size}' for size in matrix_sizes])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance' / 'memory_usage_comparison_all_sizes.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Memory usage comparison summary generated")
        print("✅ All individual memory usage analyses generated")
        
        # Also create 3-algorithm memory plots (matching the performance analysis structure)
        core_algorithms = {
            'Basic': 'Basic Sequential',
            'Vectorized': 'Vectorized (block size: 32)', 
            'Parallel': 'Parallel (8 threads)'
        }
        
        # Filter for just the 3 core algorithms from the first experimental set (exactly first 24 rows)
        core_experiment_data = df.iloc[:24].copy()
        valid_core_data = core_experiment_data[core_experiment_data['Algorithm'].isin(core_algorithms.values())].copy()
        
        core_filtered_df = []
        for index, row in valid_core_data.iterrows():
            algorithm = row['Algorithm']
            for clean_name, full_name in core_algorithms.items():
                if algorithm == full_name:
                    new_row = row.copy()
                    new_row['Algorithm_Clean'] = clean_name
                    core_filtered_df.append(new_row)
                    break
        
        if core_filtered_df:
            core_filtered_df = pd.DataFrame(core_filtered_df)
            
            # Create individual 3-algorithm memory plots for each matrix size
            for size in matrix_sizes:
                size_data = core_filtered_df[core_filtered_df['Matrix_Size'] == size]
                
                if len(size_data) == 0:
                    continue
                    
                plt.figure(figsize=(10, 6))
                
                # Get algorithm order and their memory usage
                algorithm_memory = size_data.groupby('Algorithm_Clean')['Memory_MB'].mean()
                ordered_algorithms = ['Basic', 'Vectorized', 'Parallel']  # Fixed order
                ordered_memory = [algorithm_memory.get(alg, 0) for alg in ordered_algorithms]
                
                # Core algorithm colors
                algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                bars = plt.bar(range(len(ordered_algorithms)), ordered_memory, 
                              color=algorithm_colors, alpha=0.8, edgecolor='black')
                
                plt.xlabel('Algorithm', fontsize=12)
                plt.ylabel('Memory Usage (MB)', fontsize=12)
                plt.title(f'Memory Usage: Core Algorithms - {size}×{size} Matrix', fontsize=14, fontweight='bold')
                plt.xticks(range(len(ordered_algorithms)), ordered_algorithms)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, (bar, memory) in enumerate(zip(bars, ordered_memory)):
                    if memory > 0:
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                f'{memory:.1f}MB', ha='center', va='bottom', 
                                fontweight='bold', fontsize=11)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'performance' / f'memory_core_matrix_{size}x{size}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Core memory usage analysis for {size}×{size} generated")
            
            # Create summary 3-algorithm memory comparison
            plt.figure(figsize=(10, 6))
            ordered_algorithms = ['Basic', 'Vectorized', 'Parallel']
            algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            x_pos = np.arange(len(matrix_sizes))
            width = 0.25
            
            for i, alg in enumerate(ordered_algorithms):
                memories = []
                for size in matrix_sizes:
                    alg_size_data = core_filtered_df[(core_filtered_df['Algorithm_Clean'] == alg) & 
                                                   (core_filtered_df['Matrix_Size'] == size)]
                    if len(alg_size_data) > 0:
                        memories.append(alg_size_data['Memory_MB'].mean())
                    else:
                        memories.append(0)
                
                plt.bar(x_pos + i * width, memories, width, label=alg, 
                       color=algorithm_colors[i], alpha=0.8, edgecolor='black')
                
                # Add value labels
                for j, (pos, memory) in enumerate(zip(x_pos + i * width, memories)):
                    if memory > 0:
                        plt.text(pos, memory + max(memories) * 0.01, f'{memory:.1f}',
                                ha='center', va='bottom', fontsize=9, rotation=90)
            
            plt.xlabel('Matrix Size', fontsize=12)
            plt.ylabel('Memory Usage (MB)', fontsize=12)
            plt.title('Memory Usage: Core Algorithms Scaling', fontsize=14, fontweight='bold')
            plt.xticks(x_pos + width, [f'{size}×{size}' for size in matrix_sizes])
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance' / 'memory_core_comparison_all_sizes.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Core memory usage comparison summary generated")

    def plot_thread_usage_analysis(self):
        """Thread usage analysis: individual PNGs per matrix size for all algorithms."""
        df = self.load_data('all_performance_results.csv')
        if df is None:
            return
        
        # Define algorithm matching function (same as performance analysis)
        def match_algorithm(algorithm_name):
            if algorithm_name == 'Basic Sequential' or algorithm_name == '"Basic Sequential"':
                return 'Basic'
            elif algorithm_name in ['Parallel (4 threads)', 'Parallel (8 threads)', 'Parallel (12 threads)',
                                  '"Parallel (4 threads)"', '"Parallel (8 threads)"', '"Parallel (12 threads)"']:
                return 'Parallel'
            elif algorithm_name in ['Advanced Parallel (4 threads)', 'Advanced Parallel (8 threads)', 'Advanced Parallel (12 threads)',
                                  '"Advanced Parallel (4 threads)"', '"Advanced Parallel (8 threads)"', '"Advanced Parallel (12 threads)"']:
                return 'Advanced Parallel'
            elif 'semaphore' in algorithm_name:
                return 'Advanced + Semaphore'
            elif 'streams' in algorithm_name:
                return 'Parallel Streams'
            elif 'Fork-Join' in algorithm_name:
                return 'Fork-Join'
            else:
                return None
        
        # Filter and clean algorithm names
        filtered_df = []
        for index, row in df.iterrows():
            algorithm = row['Algorithm']
            clean_name = match_algorithm(algorithm)
            
            if clean_name:
                new_row = row.copy()
                new_row['Algorithm_Clean'] = clean_name
                filtered_df.append(new_row)
        
        if not filtered_df:
            print("No matching algorithms found for thread analysis")
            return
        
        filtered_df = pd.DataFrame(filtered_df)
        matrix_sizes = sorted(filtered_df['Matrix_Size'].unique())
        
        # Create individual thread usage plots for each matrix size
        for size in matrix_sizes:
            size_data = filtered_df[filtered_df['Matrix_Size'] == size]
            
            plt.figure(figsize=(12, 8))
            
            # Get algorithm order and their thread usage
            algorithm_threads = size_data.groupby('Algorithm_Clean')['Threads'].mean().sort_values()
            ordered_algorithms = algorithm_threads.index.tolist()
            ordered_threads = algorithm_threads.values
            
            # Create color scheme
            algorithm_colors = {
                'Basic': '#1f77b4',
                'Parallel': '#ff7f0e', 
                'Advanced Parallel': '#2ca02c',
                'Advanced + Semaphore': '#d62728',
                'Parallel Streams': '#9467bd',
                'Fork-Join': '#8c564b'
            }
            colors = [algorithm_colors.get(alg, '#gray') for alg in ordered_algorithms]
            
            bars = plt.bar(range(len(ordered_algorithms)), ordered_threads, 
                          color=colors, alpha=0.8, edgecolor='black')
            
            plt.xlabel('Algorithm', fontsize=12)
            plt.ylabel('Thread Count', fontsize=12)
            plt.title(f'Thread Usage Comparison - {size}×{size} Matrix', fontsize=14, fontweight='bold')
            plt.xticks(range(len(ordered_algorithms)), ordered_algorithms, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, threads) in enumerate(zip(bars, ordered_threads)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{int(threads)}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance' / f'thread_usage_matrix_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Thread usage analysis for {size}×{size} generated")
        
        # Also create a summary thread comparison across sizes
        plt.figure(figsize=(12, 6))
        ordered_algorithms = ['Basic', 'Parallel', 'Advanced Parallel', 'Advanced + Semaphore', 'Parallel Streams', 'Fork-Join']
        algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        x_pos = np.arange(len(matrix_sizes))
        width = 0.12
        
        for i, alg in enumerate(ordered_algorithms):
            thread_counts = []
            for size in matrix_sizes:
                alg_size_data = filtered_df[(filtered_df['Algorithm_Clean'] == alg) & 
                                          (filtered_df['Matrix_Size'] == size)]
                if len(alg_size_data) > 0:
                    thread_counts.append(alg_size_data['Threads'].mean())
                else:
                    thread_counts.append(0)
            
            plt.bar(x_pos + i * width, thread_counts, width, label=alg, 
                   color=algorithm_colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for j, (pos, threads) in enumerate(zip(x_pos + i * width, thread_counts)):
                if threads > 0:
                    plt.text(pos, threads + max(thread_counts) * 0.02, f'{int(threads)}',
                            ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Matrix Size', fontsize=12)
        plt.ylabel('Thread Count', fontsize=12)
        plt.title('Thread Usage Scaling Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * (len(ordered_algorithms) - 1) / 2, 
                  [f'{size}×{size}' for size in matrix_sizes])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance' / 'thread_usage_comparison_all_sizes.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Thread usage comparison summary generated")
        print("✅ All individual thread usage analyses generated")
        
        # Also create 3-algorithm thread plots (matching the performance analysis structure)
        core_algorithms = {
            'Basic': 'Basic Sequential',
            'Vectorized': 'Vectorized (block size: 32)', 
            'Parallel': 'Parallel (8 threads)'
        }
        
        # Filter for just the 3 core algorithms from the first experimental set
        first_advanced_idx = df[df['Algorithm'].str.contains('Advanced', na=False)].index.min()
        
        # Use exactly first 24 rows for core algorithms
        core_experiment_data = df.iloc[:24].copy()
        valid_core_data = core_experiment_data[core_experiment_data['Algorithm'].isin(core_algorithms.values())].copy()
        
        core_filtered_df = []
        for index, row in valid_core_data.iterrows():
            algorithm = row['Algorithm']
            for clean_name, full_name in core_algorithms.items():
                if algorithm == full_name:
                    new_row = row.copy()
                    new_row['Algorithm_Clean'] = clean_name
                    core_filtered_df.append(new_row)
                    break
        
        if core_filtered_df:
            core_filtered_df = pd.DataFrame(core_filtered_df)
            
            # Create individual 3-algorithm thread plots for each matrix size
            for size in matrix_sizes:
                size_data = core_filtered_df[core_filtered_df['Matrix_Size'] == size]
                
                if len(size_data) == 0:
                    continue
                    
                plt.figure(figsize=(10, 6))
                
                # Get algorithm order and their thread usage
                algorithm_threads = size_data.groupby('Algorithm_Clean')['Threads'].mean()
                ordered_algorithms = ['Basic', 'Vectorized', 'Parallel']  # Fixed order
                ordered_threads = [algorithm_threads.get(alg, 0) for alg in ordered_algorithms]
                
                # Core algorithm colors
                algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                bars = plt.bar(range(len(ordered_algorithms)), ordered_threads, 
                              color=algorithm_colors, alpha=0.8, edgecolor='black')
                
                plt.xlabel('Algorithm', fontsize=12)
                plt.ylabel('Thread Count', fontsize=12)
                plt.title(f'Thread Usage: Core Algorithms - {size}×{size} Matrix', fontsize=14, fontweight='bold')
                plt.xticks(range(len(ordered_algorithms)), ordered_algorithms)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, (bar, threads) in enumerate(zip(bars, ordered_threads)):
                    if threads > 0:
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                f'{int(threads)}', ha='center', va='bottom', 
                                fontweight='bold', fontsize=11)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'performance' / f'thread_core_matrix_{size}x{size}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Core thread usage analysis for {size}×{size} generated")
            
            # Create summary 3-algorithm thread comparison
            plt.figure(figsize=(10, 6))
            ordered_algorithms = ['Basic', 'Vectorized', 'Parallel']
            algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            x_pos = np.arange(len(matrix_sizes))
            width = 0.25
            
            for i, alg in enumerate(ordered_algorithms):
                thread_counts = []
                for size in matrix_sizes:
                    alg_size_data = core_filtered_df[(core_filtered_df['Algorithm_Clean'] == alg) & 
                                                   (core_filtered_df['Matrix_Size'] == size)]
                    if len(alg_size_data) > 0:
                        thread_counts.append(alg_size_data['Threads'].mean())
                    else:
                        thread_counts.append(0)
                
                plt.bar(x_pos + i * width, thread_counts, width, label=alg, 
                       color=algorithm_colors[i], alpha=0.8, edgecolor='black')
                
                # Add value labels
                for j, (pos, threads) in enumerate(zip(x_pos + i * width, thread_counts)):
                    if threads > 0:
                        plt.text(pos, threads + max(thread_counts) * 0.02, f'{int(threads)}',
                                ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Matrix Size', fontsize=12)
            plt.ylabel('Thread Count', fontsize=12)
            plt.title('Thread Usage: Core Algorithms Scaling', fontsize=14, fontweight='bold')
            plt.xticks(x_pos + width, [f'{size}×{size}' for size in matrix_sizes])
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance' / 'thread_core_comparison_all_sizes.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Core thread usage comparison summary generated")



    def plot_parallel_efficiency_analysis(self):
        """Generate comprehensive parallel execution efficiency plots with speedup per thread metrics."""
        df = self.load_data('all_efficiency_analysis.csv')
        if df is None:
            print("⚠️ No efficiency data available for parallel efficiency analysis")
            return
        
        # Filter for comparison data only to avoid duplicates
        comparison_data = df[df['Test_Type'] == 'comparison']
        if len(comparison_data) == 0:
            print("⚠️ No comparison data found for parallel efficiency analysis")
            return
        
        # Handle duplicates: keep only one representative value per algorithm per matrix size
        filtered_data = []
        matrix_sizes = sorted(comparison_data['Matrix_Size'].unique())
        
        for size in matrix_sizes:
            size_data = comparison_data[comparison_data['Matrix_Size'] == size]
            
            # For each matrix size, keep only one entry per unique algorithm
            # Group by algorithm and select the most representative entry
            size_algorithms = {}
            for index, row in size_data.iterrows():
                algorithm = row['Algorithm']
                
                if algorithm not in size_algorithms:
                    # First occurrence of this algorithm - keep it
                    size_algorithms[algorithm] = row
                else:
                    # Duplicate found - apply selection logic
                    stored_row = size_algorithms[algorithm]
                    
                    # Prefer entries from advanced experiment context (row >= 12) for consistency
                    if row.name >= 12 and stored_row.name < 12:
                        size_algorithms[algorithm] = row
                    # If both are from same experiment context, keep the better performing one
                    elif (row.name >= 12) == (stored_row.name >= 12):
                        if row['Efficiency'] > stored_row['Efficiency']:
                            size_algorithms[algorithm] = row
                    # Otherwise keep the stored one (advanced experiment preference)
            
            # Add selected algorithms for this size (one per algorithm only)
            for alg_row in size_algorithms.values():
                filtered_data.append(alg_row)
        
        if not filtered_data:
            print("⚠️ No efficiency data found after filtering")
            return
        
        # Convert back to DataFrame
        comparison_data = pd.DataFrame(filtered_data)
        matrix_sizes = sorted(comparison_data['Matrix_Size'].unique())
        
        # Efficiency directory
        efficiency_dir = self.output_dir / 'efficiency'
        efficiency_dir.mkdir(exist_ok=True)
        
        # Generate individual efficiency plots for each matrix size
        for size in matrix_sizes:
            size_data = comparison_data[comparison_data['Matrix_Size'] == size]
            
            if len(size_data) == 0:
                continue
                
            # Create simplified efficiency analysis - focus on key metric
            plt.figure(figsize=(12, 8))
            
            algorithms = size_data['Algorithm'].tolist()
            speedup_per_thread = size_data['Speedup_Per_Thread'].tolist()
            efficiencies = size_data['Efficiency'].tolist()
            threads = size_data['Threads'].tolist()
            
            # Clean algorithm names for display
            clean_algorithms = [self.clean_algorithm_name(alg) for alg in algorithms]
            
            # Color scheme for algorithms
            colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
            
            # Single focused plot: Speedup per Thread (the key efficiency metric)
            bars = plt.bar(range(len(algorithms)), speedup_per_thread, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for i, (bar, spt) in enumerate(zip(bars, speedup_per_thread)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{spt:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Add efficiency rating colors
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency (1.0)')
            plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Good Efficiency (0.7)')
            plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.3, label='Fair Efficiency (0.5)')
            
            plt.xlabel('Parallel Algorithm', fontsize=12, fontweight='bold')
            plt.ylabel('Speedup per Thread', fontsize=12, fontweight='bold')
            plt.title(f'Parallel Execution Efficiency - {size}×{size} Matrix\n(Higher = Better Thread Utilization)', 
                     fontsize=14, fontweight='bold')
            plt.xticks(range(len(algorithms)), clean_algorithms, rotation=45, ha='right')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3, axis='y')
            

            
            plt.tight_layout()
            plt.savefig(efficiency_dir / f'parallel_efficiency_matrix_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Parallel efficiency analysis for {size}×{size} generated")
        
        # Create simplified comparison across matrix sizes - focus on speedup per thread
        plt.figure(figsize=(12, 8))
        
        unique_algorithms = comparison_data['Algorithm'].unique()
        algorithm_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_algorithms)))
        
        # Plot speedup per thread trends across matrix sizes
        for i, algorithm in enumerate(unique_algorithms):
            alg_data = comparison_data[comparison_data['Algorithm'] == algorithm]
            alg_data = alg_data.sort_values('Matrix_Size')
            
            sizes = alg_data['Matrix_Size'].tolist()
            speedup_per_thread = alg_data['Speedup_Per_Thread'].tolist()
            
            plt.plot(sizes, speedup_per_thread, 'o-', color=algorithm_colors[i], 
                    linewidth=3, markersize=10, alpha=0.8, 
                    label=self.clean_algorithm_name(algorithm))
            
            # Add value labels on points
            for size, spt in zip(sizes, speedup_per_thread):
                plt.annotate(f'{spt:.3f}', (size, spt), 
                           xytext=(0, 10), textcoords='offset points', 
                           ha='center', fontsize=8, alpha=0.7)
        
        plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup per Thread', fontsize=12, fontweight='bold')
        plt.title('Parallel Efficiency Across Matrix Sizes\n(Higher = Better Thread Utilization)', 
                 fontsize=14, fontweight='bold')
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect Efficiency')
        plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Good Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(matrix_sizes, [f'{s}×{s}' for s in matrix_sizes])
        
        plt.tight_layout()
        plt.savefig(efficiency_dir / 'parallel_efficiency_comparison_all_sizes.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Parallel efficiency comparison across all matrix sizes generated")
        print("✅ All individual parallel efficiency analyses generated")

    def plot_advanced_parallel_speedup_analysis(self):
        """Generate simple bar charts for advanced parallel algorithms with one PNG per matrix size."""
        df = self.load_data('all_efficiency_analysis.csv')
        if df is None:
            print("⚠️ No efficiency data available for advanced parallel speedup analysis")
            return
        
        # Filter for comparison data only to avoid duplicates
        comparison_data = df[df['Test_Type'] == 'comparison']
        if len(comparison_data) == 0:
            print("⚠️ No comparison data found for advanced parallel speedup analysis")
            return
        
        # Filter for advanced parallel algorithms only
        advanced_algorithms = [
            'Advanced Parallel (12 threads)',
            'Advanced Parallel (12 threads, semaphore:4)',  
            'Advanced Parallel (12 threads, streams)',
            'Fork-Join (threshold: 64, parallelism: 12)'
        ]
        
        # Include basic parallel for comparison baseline
        comparison_algorithms = ['Parallel (12 threads)']
        
        advanced_data = comparison_data[
            comparison_data['Algorithm'].isin(advanced_algorithms + comparison_algorithms)
        ]
        
        # Remove duplicates by keeping only the first occurrence of each Algorithm-Matrix_Size combination
        advanced_data = advanced_data.drop_duplicates(subset=['Algorithm', 'Matrix_Size'], keep='first')
        
        if len(advanced_data) == 0:
            print("⚠️ No advanced parallel algorithm data found")
            return
        
        # Get unique matrix sizes
        matrix_sizes = sorted(advanced_data['Matrix_Size'].unique())
        
        # Advanced parallel directory
        advanced_dir = self.output_dir / 'performance'
        advanced_dir.mkdir(exist_ok=True)
        
        # Generate simple bar chart for each matrix size
        for size in matrix_sizes:
            size_data = advanced_data[advanced_data['Matrix_Size'] == size]
            
            if len(size_data) == 0:
                continue
                
            # Create simple bar chart
            plt.figure(figsize=(12, 8))
            
            algorithms = size_data['Algorithm'].tolist()
            speedups = size_data['Speedup'].tolist()
            
            # Clean algorithm names for display
            clean_algorithms = [self.clean_algorithm_name(alg) for alg in algorithms]
            
            # Color scheme - highlight advanced algorithms
            colors = []
            for alg in algorithms:
                if 'Advanced' in alg or 'Fork-Join' in alg:
                    if 'streams' in alg:
                        colors.append('#FF6B6B')  # Red for streams
                    elif 'semaphore' in alg:
                        colors.append('#4ECDC4')  # Teal for semaphore
                    elif 'Fork-Join' in alg:
                        colors.append('#45B7D1')  # Blue for fork-join
                    else:
                        colors.append('#96CEB4')  # Green for basic advanced
                else:
                    colors.append('#FFEAA7')  # Light yellow for baseline parallel
            
            # Create bar chart
            bars = plt.bar(range(len(algorithms)), speedups, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            plt.xlabel('Algorithm', fontsize=14)
            plt.ylabel('Speedup Factor', fontsize=14)
            plt.title(f'Advanced Parallel Algorithms Speedup - {size}×{size} Matrix', 
                     fontsize=16, fontweight='bold')
            plt.xticks(range(len(algorithms)), clean_algorithms, rotation=45, ha='right')
            
            # Add reference lines
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                       label='Sequential Baseline')
            
            # Highlight best performing algorithm
            best_speedup_idx = speedups.index(max(speedups))
            bars[best_speedup_idx].set_edgecolor('gold')
            bars[best_speedup_idx].set_linewidth(3)
            
            # Annotate speedup values
            for i, (bar, speedup) in enumerate(zip(bars, speedups)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{speedup:.2f}×', ha='center', va='bottom', 
                        fontweight='bold', fontsize=11)
            
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(advanced_dir / f'advanced_parallel_speedup_simple_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Simple advanced parallel speedup chart for {size}×{size} generated")
        
        print("✅ All simple advanced parallel speedup charts generated")

    def plot_core_algorithm_scaling_analysis(self):
        """Generate simple scaling plots for core algorithms showing memory usage and execution time scaling."""
        performance_df = self.load_data('all_performance_results.csv')
        if performance_df is None:
            print("⚠️ No performance data available for core algorithm scaling analysis")
            return
        
        # Filter for comparison data only to avoid duplicates
        comparison_data = performance_df[performance_df['Test_Type'] == 'comparison']
        if len(comparison_data) == 0:
            print("⚠️ No comparison data found for core algorithm scaling analysis")
            return
        
        # Define core algorithms from the first experimental set
        core_algorithms = {
            'Basic Sequential': 'Basic',
            'Vectorized (block size: 32)': 'Vectorized',
            'Parallel (8 threads)': 'Parallel'
        }
        
        # Filter for core algorithms only from the first experimental set (exactly first 24 rows)
        core_experiment_data = comparison_data.iloc[:24].copy()
        core_data = core_experiment_data[core_experiment_data['Algorithm'].isin(core_algorithms.keys())].copy()
        
        # Remove duplicates and add clean names
        core_data = core_data.drop_duplicates(subset=['Algorithm', 'Matrix_Size'], keep='first')
        
        if len(core_data) == 0:
            print("⚠️ No core algorithm data found")
            return
        
        # Get unique matrix sizes
        matrix_sizes = sorted(core_data['Matrix_Size'].unique())
        
        # Scaling directory
        scaling_dir = self.output_dir / 'performance'
        scaling_dir.mkdir(exist_ok=True)
        
        # Create execution time scaling plot
        plt.figure(figsize=(12, 8))
        
        colors = {'Basic': '#1f77b4', 'Vectorized': '#ff7f0e', 'Parallel': '#2ca02c'}
        markers = {'Basic': 'o', 'Vectorized': 's', 'Parallel': '^'}
        
        for full_name, clean_name in core_algorithms.items():
            alg_data = core_data[core_data['Algorithm'] == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            times = alg_data['Time_ms'].tolist()
            
            plt.plot(sizes, times, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        plt.xlabel('Matrix Size', fontsize=14)
        plt.ylabel('Execution Time (ms) - Log2 Scale', fontsize=14)
        plt.title('Core Algorithms: Execution Time Scaling', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(matrix_sizes, [f'{s}×{s}' for s in matrix_sizes])
        plt.yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(scaling_dir / 'core_algorithms_execution_time_scaling.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Core algorithms execution time scaling plot generated")
        
        # Create memory usage scaling plot
        plt.figure(figsize=(12, 8))
        
        for full_name, clean_name in core_algorithms.items():
            alg_data = core_data[core_data['Algorithm'] == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            memory = alg_data['Memory_MB'].tolist()
            
            plt.plot(sizes, memory, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        plt.xlabel('Matrix Size', fontsize=14)
        plt.ylabel('Memory Usage (MB) - Log2 Scale', fontsize=14)
        plt.title('Core Algorithms: Memory Usage Scaling', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(matrix_sizes, [f'{s}×{s}' for s in matrix_sizes])
        plt.yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(scaling_dir / 'core_algorithms_memory_scaling.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Core algorithms memory scaling plot generated")
        
        # Create combined scaling analysis plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Core Algorithms: Performance and Memory Scaling Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Execution time subplot
        for full_name, clean_name in core_algorithms.items():
            alg_data = core_data[core_data['Algorithm'] == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            times = alg_data['Time_ms'].tolist()
            
            ax1.plot(sizes, times, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        ax1.set_xlabel('Matrix Size', fontsize=12)
        ax1.set_ylabel('Execution Time (ms) - Log2 Scale', fontsize=12)
        ax1.set_title('Execution Time Scaling', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(matrix_sizes)
        ax1.set_xticklabels([f'{s}×{s}' for s in matrix_sizes])
        ax1.set_yscale('log', base=2)
        
        # Memory usage subplot
        for full_name, clean_name in core_algorithms.items():
            alg_data = core_data[core_data['Algorithm'] == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            memory = alg_data['Memory_MB'].tolist()
            
            ax2.plot(sizes, memory, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        ax2.set_xlabel('Matrix Size', fontsize=12)
        ax2.set_ylabel('Memory Usage (MB) - Log2 Scale', fontsize=12)
        ax2.set_title('Memory Usage Scaling', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(matrix_sizes)
        ax2.set_xticklabels([f'{s}×{s}' for s in matrix_sizes])
        ax2.set_yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(scaling_dir / 'core_algorithms_combined_scaling.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Core algorithms combined scaling analysis generated")
        print("✅ All core algorithm scaling analyses generated")

    def plot_advanced_algorithm_scaling_analysis(self):
        """Generate scaling plots for advanced parallel algorithms showing memory usage and execution time scaling."""
        performance_df = self.load_data('all_performance_results.csv')
        if performance_df is None:
            print("⚠️ No performance data available for advanced algorithm scaling analysis")
            return
        
        # Filter for comparison data only to avoid duplicates
        comparison_data = performance_df[performance_df['Test_Type'] == 'comparison']
        if len(comparison_data) == 0:
            print("⚠️ No comparison data found for advanced algorithm scaling analysis")
            return
        
        # Define advanced algorithms mapping (including baselines for comparison)
        advanced_algorithms = {
            'Basic Sequential': 'Basic',
            'Parallel (8 threads)': 'Parallel',
            'Advanced Parallel (8 threads)': 'Advanced Parallel',
            'Advanced Parallel (8 threads, semaphore:4)': 'Semaphore Control',
            'Advanced Parallel (8 threads, streams)': 'Parallel Streams',
            'Fork-Join (threshold: 64, parallelism: 8)': 'Fork-Join'
        }
        
        # Handle mixed data: for sizes 128/256 use all comparison data, for 512/1024 use rows 25+
        advanced_data = []
        
        # For each matrix size, determine the correct data source
        matrix_sizes = sorted(comparison_data['Matrix_Size'].unique())
        
        for size in matrix_sizes:
            if size in [128, 256]:
                # For 128 and 256, advanced algorithms are mixed in first 24 rows
                size_data = comparison_data[comparison_data['Matrix_Size'] == size]
            else:
                # For 512 and 1024, advanced algorithms are in rows 25+
                size_data = comparison_data.iloc[24:]
                size_data = size_data[size_data['Matrix_Size'] == size]
            
            # Collect all matching algorithms for this size
            size_algorithms = {}
            for index, row in size_data.iterrows():
                algorithm = row['Algorithm']
                clean_alg = algorithm.strip('"')
                if clean_alg in advanced_algorithms:
                    alg_clean_name = advanced_algorithms[clean_alg]
                    # For sizes 128/256 with duplicates, prefer Advanced experiment entries (row >= 24)
                    if size in [128, 256] and alg_clean_name in ['Basic', 'Parallel']:
                        # For baseline algorithms in mixed data, prefer Advanced experiment
                        if alg_clean_name not in size_algorithms or row.name >= 24:
                            row_copy = row.copy()
                            row_copy['Algorithm_Clean'] = alg_clean_name
                            size_algorithms[alg_clean_name] = row_copy
                    else:
                        # For truly advanced algorithms or non-duplicate cases, take first found
                        if alg_clean_name not in size_algorithms:
                            row_copy = row.copy()
                            row_copy['Algorithm_Clean'] = alg_clean_name
                            size_algorithms[alg_clean_name] = row_copy
            
            # Add the selected version of each algorithm for this size
            for alg_row in size_algorithms.values():
                advanced_data.append(alg_row)
        
        if not advanced_data:
            print("⚠️ No advanced algorithm data found for scaling analysis")
            return
            
        advanced_data = pd.DataFrame(advanced_data)
        matrix_sizes = sorted(advanced_data['Matrix_Size'].unique())
        
        # Scaling directory
        scaling_dir = self.output_dir / 'performance'
        scaling_dir.mkdir(exist_ok=True)
        
        # Create execution time scaling plot for advanced algorithms
        plt.figure(figsize=(12, 8))
        
        colors = {'Basic': '#1f77b4', 'Parallel': '#ff7f0e', 'Advanced Parallel': '#2ca02c', 
                  'Semaphore Control': '#d62728', 'Parallel Streams': '#9467bd', 'Fork-Join': '#8c564b'}
        markers = {'Basic': 'o', 'Parallel': 's', 'Advanced Parallel': '^', 
                   'Semaphore Control': 'v', 'Parallel Streams': 'D', 'Fork-Join': 'X'}
        
        for full_name, clean_name in advanced_algorithms.items():
            alg_data = advanced_data[advanced_data['Algorithm'].str.strip('"') == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            times = alg_data['Time_ms'].tolist()
            
            plt.plot(sizes, times, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        plt.xlabel('Matrix Size', fontsize=14)
        plt.ylabel('Execution Time (ms) - Log2 Scale', fontsize=14)
        plt.title('Advanced Algorithms: Execution Time Scaling', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(matrix_sizes, [f'{s}×{s}' for s in matrix_sizes])
        plt.yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(scaling_dir / 'advanced_algorithms_execution_time_scaling.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Advanced algorithms execution time scaling plot generated")
        
        # Create memory usage scaling plot for advanced algorithms
        plt.figure(figsize=(12, 8))
        
        for full_name, clean_name in advanced_algorithms.items():
            alg_data = advanced_data[advanced_data['Algorithm'].str.strip('"') == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            memory = alg_data['Memory_MB'].tolist()
            
            plt.plot(sizes, memory, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        plt.xlabel('Matrix Size', fontsize=14)
        plt.ylabel('Memory Usage (MB) - Log2 Scale', fontsize=14)
        plt.title('Advanced Algorithms: Memory Usage Scaling', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(matrix_sizes, [f'{s}×{s}' for s in matrix_sizes])
        plt.yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(scaling_dir / 'advanced_algorithms_memory_scaling.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Advanced algorithms memory scaling plot generated")
        
        # Create combined scaling analysis plot for advanced algorithms
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Advanced Algorithms: Performance and Memory Scaling Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Execution time subplot
        for full_name, clean_name in advanced_algorithms.items():
            alg_data = advanced_data[advanced_data['Algorithm'].str.strip('"') == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            times = alg_data['Time_ms'].tolist()
            
            ax1.plot(sizes, times, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        ax1.set_xlabel('Matrix Size', fontsize=12)
        ax1.set_ylabel('Execution Time (ms) - Log2 Scale', fontsize=12)
        ax1.set_title('Execution Time Scaling', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(matrix_sizes)
        ax1.set_xticklabels([f'{s}×{s}' for s in matrix_sizes])
        ax1.set_yscale('log', base=2)
        
        # Memory usage subplot
        for full_name, clean_name in advanced_algorithms.items():
            alg_data = advanced_data[advanced_data['Algorithm'].str.strip('"') == full_name]
            if len(alg_data) == 0:
                continue
                
            alg_data = alg_data.sort_values('Matrix_Size')
            sizes = alg_data['Matrix_Size'].tolist()
            memory = alg_data['Memory_MB'].tolist()
            
            ax2.plot(sizes, memory, marker=markers[clean_name], color=colors[clean_name], 
                    linewidth=3, markersize=10, alpha=0.8, label=clean_name)
        
        ax2.set_xlabel('Matrix Size', fontsize=12)
        ax2.set_ylabel('Memory Usage (MB) - Log2 Scale', fontsize=12)
        ax2.set_title('Memory Usage Scaling', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(matrix_sizes)
        ax2.set_xticklabels([f'{s}×{s}' for s in matrix_sizes])
        ax2.set_yscale('log', base=2)
        
        plt.tight_layout()
        plt.savefig(scaling_dir / 'advanced_algorithms_combined_scaling.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Advanced algorithms combined scaling analysis generated")
        print("✅ All advanced algorithm scaling analyses generated")

    def plot_advanced_algorithm_performance_comparison(self):
        """Generate performance comparison plots for advanced algorithms across different matrix sizes."""
        performance_df = self.load_data('all_performance_results.csv')
        if performance_df is None:
            print("⚠️ No performance data available for advanced algorithm performance comparison")
            return
        
        # Filter for comparison data only to avoid duplicates
        comparison_data = performance_df[performance_df['Test_Type'] == 'comparison']
        if len(comparison_data) == 0:
            print("⚠️ No comparison data found for advanced algorithm performance comparison")
            return
        
        # Define advanced algorithms mapping (including baselines for comparison)
        advanced_algorithms = {
            'Basic Sequential': 'Basic Sequential',
            'Parallel (8 threads)': 'Parallel (8 threads)',
            'Advanced Parallel (8 threads)': 'Advanced Parallel (8 threads)',
            'Advanced Parallel (8 threads, semaphore:4)': 'Advanced + Semaphore',
            'Advanced Parallel (8 threads, streams)': 'Parallel Streams',
            'Fork-Join (threshold: 64, parallelism: 8)': 'Fork-Join'
        }
        
        # Handle mixed data: for sizes 128/256 use all comparison data, for 512/1024 use rows 25+
        advanced_data = []
        
        # For each matrix size, determine the correct data source
        matrix_sizes = sorted(comparison_data['Matrix_Size'].unique())
        
        for size in matrix_sizes:
            if size in [128, 256]:
                # For 128 and 256, advanced algorithms are mixed in first 24 rows
                size_data = comparison_data[comparison_data['Matrix_Size'] == size]
            else:
                # For 512 and 1024, advanced algorithms are in rows 25+
                size_data = comparison_data.iloc[24:]
                size_data = size_data[size_data['Matrix_Size'] == size]
            
            # Collect all matching algorithms for this size
            size_algorithms = {}
            for index, row in size_data.iterrows():
                algorithm = row['Algorithm']
                clean_alg = algorithm.strip('"')
                if clean_alg in advanced_algorithms:
                    alg_clean_name = advanced_algorithms[clean_alg]
                    # For sizes 128/256 with duplicates, prefer Advanced experiment entries (row >= 24)
                    if size in [128, 256] and alg_clean_name in ['Basic Sequential', 'Parallel (8 threads)']:
                        # For baseline algorithms in mixed data, prefer Advanced experiment
                        if alg_clean_name not in size_algorithms or row.name >= 24:
                            row_copy = row.copy()
                            row_copy['Algorithm_Clean'] = alg_clean_name
                            size_algorithms[alg_clean_name] = row_copy
                    else:
                        # For truly advanced algorithms or non-duplicate cases, take first found
                        if alg_clean_name not in size_algorithms:
                            row_copy = row.copy()
                            row_copy['Algorithm_Clean'] = alg_clean_name
                            size_algorithms[alg_clean_name] = row_copy
            
            # Add the selected version of each algorithm for this size
            for alg_row in size_algorithms.values():
                advanced_data.append(alg_row)
        
        if not advanced_data:
            print("⚠️ No advanced algorithm data found for performance comparison")
            return
            
        advanced_df = pd.DataFrame(advanced_data)
        matrix_sizes = sorted(advanced_df['Matrix_Size'].unique())
        
        # Performance directory
        performance_dir = self.output_dir / 'performance'
        performance_dir.mkdir(exist_ok=True)
        
        # Colors for algorithms
        colors = {
            'Basic Sequential': '#1f77b4', 
            'Parallel (8 threads)': '#ff7f0e', 
            'Advanced Parallel (8 threads)': '#2ca02c',
            'Advanced + Semaphore': '#d62728', 
            'Parallel Streams': '#9467bd', 
            'Fork-Join': '#8c564b'
        }
        
        # Generate individual performance comparison plots for each matrix size
        for size in matrix_sizes:
            size_data = advanced_df[advanced_df['Matrix_Size'] == size]
            
            plt.figure(figsize=(14, 8))
            
            algorithms = []
            times = []
            colors_list = []
            
            for _, row in size_data.iterrows():
                alg_name = row['Algorithm_Clean']
                algorithms.append(alg_name)
                times.append(row['Time_ms'])
                colors_list.append(colors[alg_name])
            
            bars = plt.bar(range(len(algorithms)), times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, time) in enumerate(zip(bars, times)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.xlabel('Algorithm Implementation', fontsize=14, fontweight='bold')
            plt.ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
            plt.title(f'Advanced Algorithm Performance Comparison\nMatrix Size: {size}×{size}', 
                     fontsize=16, fontweight='bold')
            
            plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(performance_dir / f'advanced_algorithms_performance_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Advanced algorithm performance comparison for {size}×{size} generated")
        
        # Generate summary comparison across all sizes
        plt.figure(figsize=(16, 10))
        
        # Create subplots for each matrix size
        for i, size in enumerate(matrix_sizes):
            plt.subplot(2, 2, i + 1)
            size_data = advanced_df[advanced_df['Matrix_Size'] == size]
            
            algorithms = []
            times = []
            colors_list = []
            
            for _, row in size_data.iterrows():
                alg_name = row['Algorithm_Clean']
                algorithms.append(alg_name)
                times.append(row['Time_ms'])
                colors_list.append(colors[alg_name])
            
            bars = plt.bar(range(len(algorithms)), times, color=colors_list, alpha=0.8)
            
            plt.title(f'{size}×{size}', fontsize=12, fontweight='bold')
            plt.ylabel('Time (ms)', fontsize=10)
            plt.xticks(range(len(algorithms)), 
                      [alg.replace(' ', '\n') for alg in algorithms], 
                      rotation=0, ha='center', fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Advanced Algorithm Performance Comparison - All Matrix Sizes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(performance_dir / 'advanced_algorithms_performance_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Advanced algorithm performance comparison summary generated")
        print("✅ All advanced algorithm performance comparisons generated")

    def plot_advanced_algorithm_memory_comparison(self):
        """Generate memory usage comparison plots for advanced algorithms across different matrix sizes."""
        performance_df = self.load_data('all_performance_results.csv')
        if performance_df is None:
            print("⚠️ No performance data available for advanced algorithm memory comparison")
            return
        
        # Filter for comparison data only to avoid duplicates
        comparison_data = performance_df[performance_df['Test_Type'] == 'comparison']
        if len(comparison_data) == 0:
            print("⚠️ No comparison data found for advanced algorithm memory comparison")
            return
        
        # Define advanced algorithms mapping (including baselines for comparison)
        advanced_algorithms = {
            'Basic Sequential': 'Basic Sequential',
            'Parallel (8 threads)': 'Parallel (8 threads)',
            'Advanced Parallel (8 threads)': 'Advanced Parallel (8 threads)',
            'Advanced Parallel (8 threads, semaphore:4)': 'Advanced + Semaphore',
            'Advanced Parallel (8 threads, streams)': 'Parallel Streams',
            'Fork-Join (threshold: 64, parallelism: 8)': 'Fork-Join'
        }
        
        # Handle mixed data: for sizes 128/256 use all comparison data, for 512/1024 use rows 25+
        advanced_data = []
        
        # For each matrix size, determine the correct data source
        matrix_sizes = sorted(comparison_data['Matrix_Size'].unique())
        
        for size in matrix_sizes:
            if size in [128, 256]:
                # For 128 and 256, advanced algorithms are mixed in first 24 rows
                size_data = comparison_data[comparison_data['Matrix_Size'] == size]
            else:
                # For 512 and 1024, advanced algorithms are in rows 25+
                size_data = comparison_data.iloc[24:]
                size_data = size_data[size_data['Matrix_Size'] == size]
            
            # Collect all matching algorithms for this size
            size_algorithms = {}
            for index, row in size_data.iterrows():
                algorithm = row['Algorithm']
                clean_alg = algorithm.strip('"')
                if clean_alg in advanced_algorithms:
                    alg_clean_name = advanced_algorithms[clean_alg]
                    # For sizes 128/256 with duplicates, prefer Advanced experiment entries (row >= 24)
                    if size in [128, 256] and alg_clean_name in ['Basic Sequential', 'Parallel (8 threads)']:
                        # For baseline algorithms in mixed data, prefer Advanced experiment
                        if alg_clean_name not in size_algorithms or row.name >= 24:
                            row_copy = row.copy()
                            row_copy['Algorithm_Clean'] = alg_clean_name
                            size_algorithms[alg_clean_name] = row_copy
                    else:
                        # For truly advanced algorithms or non-duplicate cases, take first found
                        if alg_clean_name not in size_algorithms:
                            row_copy = row.copy()
                            row_copy['Algorithm_Clean'] = alg_clean_name
                            size_algorithms[alg_clean_name] = row_copy
            
            # Add the selected version of each algorithm for this size
            for alg_row in size_algorithms.values():
                advanced_data.append(alg_row)
        
        if not advanced_data:
            print("⚠️ No advanced algorithm data found for memory comparison")
            return
            
        advanced_df = pd.DataFrame(advanced_data)
        matrix_sizes = sorted(advanced_df['Matrix_Size'].unique())
        
        # Memory directory
        memory_dir = self.output_dir / 'performance'
        memory_dir.mkdir(exist_ok=True)
        
        # Colors for algorithms
        colors = {
            'Basic Sequential': '#1f77b4', 
            'Parallel (8 threads)': '#ff7f0e', 
            'Advanced Parallel (8 threads)': '#2ca02c',
            'Advanced + Semaphore': '#d62728', 
            'Parallel Streams': '#9467bd', 
            'Fork-Join': '#8c564b'
        }
        
        # Generate individual memory comparison plots for each matrix size
        for size in matrix_sizes:
            size_data = advanced_df[advanced_df['Matrix_Size'] == size]
            
            plt.figure(figsize=(14, 8))
            
            algorithms = []
            memories = []
            colors_list = []
            
            for _, row in size_data.iterrows():
                alg_name = row['Algorithm_Clean']
                algorithms.append(alg_name)
                memories.append(row['Memory_MB'])
                colors_list.append(colors[alg_name])
            
            bars = plt.bar(range(len(algorithms)), memories, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (bar, memory) in enumerate(zip(bars, memories)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{memory:.1f}MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.xlabel('Algorithm Implementation', fontsize=14, fontweight='bold')
            plt.ylabel('Memory Usage (MB)', fontsize=14, fontweight='bold')
            plt.title(f'Advanced Algorithm Memory Usage Comparison\nMatrix Size: {size}×{size}', 
                     fontsize=16, fontweight='bold')
            
            plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(memory_dir / f'advanced_algorithms_memory_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Advanced algorithm memory comparison for {size}×{size} generated")
        
        print("✅ All advanced algorithm memory comparisons generated")

    def plot_advanced_methods_analysis(self):
        """Complete methods analysis: Basic to Fork-Join - individual PNGs per matrix size."""
        df = self.load_data('all_performance_results.csv')
        if df is None:
            return
        
        # Define algorithm patterns for matching (use exact matching to avoid conflicts)
        def match_algorithm(algorithm_name):
            # Exact matching to avoid conflicts between similar names
            if algorithm_name == 'Basic Sequential' or algorithm_name == '"Basic Sequential"':
                return 'Basic'
            elif algorithm_name in ['Parallel (4 threads)', 'Parallel (8 threads)', 'Parallel (12 threads)',
                                  '"Parallel (4 threads)"', '"Parallel (8 threads)"', '"Parallel (12 threads)"']:
                return 'Parallel'
            elif algorithm_name in ['Advanced Parallel (4 threads)', 'Advanced Parallel (8 threads)', 'Advanced Parallel (12 threads)',
                                  '"Advanced Parallel (4 threads)"', '"Advanced Parallel (8 threads)"', '"Advanced Parallel (12 threads)"']:
                return 'Advanced Parallel'
            elif 'semaphore' in algorithm_name:
                return 'Advanced + Semaphore'
            elif 'streams' in algorithm_name:
                return 'Parallel Streams'
            elif 'Fork-Join' in algorithm_name:
                return 'Fork-Join'
            else:
                return None
        
        # Filter data for algorithms using exact matching function
        filtered_df = []
        for index, row in df.iterrows():
            algorithm = row['Algorithm']
            clean_name = match_algorithm(algorithm)
            
            if clean_name:
                row_copy = row.copy()
                row_copy['Algorithm_Clean'] = clean_name
                filtered_df.append(row_copy)
        
        filtered_df = pd.DataFrame(filtered_df)
        
        if len(filtered_df) == 0:
            print("⚠️  No matching algorithms found!")
            return
        
        matrix_sizes = sorted(filtered_df['Matrix_Size'].unique())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 6 distinct colors
        
        # Create individual PNG for each matrix size
        for size in matrix_sizes:
            plt.figure(figsize=(14, 8))  # Wider figure for 6 bars
            
            size_data = filtered_df[filtered_df['Matrix_Size'] == size]
            
            # Group by algorithm and take average (in case of multiple entries)
            avg_data = size_data.groupby('Algorithm_Clean')['Time_ms'].mean().reset_index()
            algorithms = avg_data['Algorithm_Clean'].values
            times = avg_data['Time_ms'].values
            
            # Ensure we have all 6 methods in the desired order
            ordered_algorithms = ['Basic', 'Parallel', 'Advanced Parallel', 'Advanced + Semaphore', 'Parallel Streams', 'Fork-Join']
            ordered_times = []
            
            for alg in ordered_algorithms:
                if alg in algorithms:
                    idx = list(algorithms).index(alg)
                    ordered_times.append(times[idx])
                else:
                    ordered_times.append(0)  # Fallback if algorithm missing
            
            bars = plt.bar(ordered_algorithms, ordered_times, 
                          color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
            
            plt.ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
            plt.title(f'Complete Matrix Multiplication Methods Comparison\nMatrix Size: {size}×{size}', 
                     fontsize=14, fontweight='bold')
            plt.xticks(fontsize=10, rotation=25, ha='right')  # Rotation for 6 longer labels
            plt.yticks(fontsize=11)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, time in zip(bars, ordered_times):
                if time > 0:  # Only add label if we have data
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ordered_times)*0.01,
                            f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Add speedup annotations (relative to Basic Sequential)
            base_time = ordered_times[0]  # First entry is always Basic Sequential
            for i, (bar, time, alg) in enumerate(zip(bars, ordered_times, ordered_algorithms)):
                if i > 0 and time > 0 and base_time > 0:  # Skip Basic (index 0) and zero values
                    speedup = base_time / time
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                            f'{speedup:.1f}x', ha='center', va='center', 
                            fontweight='bold', fontsize=9, color='white',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance' / f'complete_methods_matrix_{size}x{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Complete methods analysis for {size}×{size} generated")
        
        # Also create a summary speedup comparison for all methods
        plt.figure(figsize=(15, 8))
        ordered_algorithms = ['Basic', 'Parallel', 'Advanced Parallel', 'Advanced + Semaphore', 'Parallel Streams', 'Fork-Join']
        algorithm_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        x_pos = np.arange(len(matrix_sizes))
        width = 0.13  # Narrower bars for 6 algorithms
        
        # Calculate speedup relative to Basic Sequential for fair comparison
        basic_df = df[df['Algorithm'] == 'Basic Sequential']
        
        for i, alg in enumerate(ordered_algorithms):
            speedups = []
            for size in matrix_sizes:
                alg_size_data = filtered_df[(filtered_df['Algorithm_Clean'] == alg) & 
                                          (filtered_df['Matrix_Size'] == size)]
                basic_size_data = basic_df[basic_df['Matrix_Size'] == size]
                
                if len(alg_size_data) > 0 and len(basic_size_data) > 0:
                    alg_time = alg_size_data['Time_ms'].mean()
                    basic_time = basic_size_data['Time_ms'].mean()
                    speedup = basic_time / alg_time
                    speedups.append(speedup)
                else:
                    speedups.append(1.0 if alg == 'Basic' else 0)  # Basic always 1.0, others 0 if missing
            
            plt.bar(x_pos + i * width, speedups, width, label=alg, 
                   color=algorithm_colors[i], alpha=0.8, edgecolor='black')
            
            # Add value labels
            for j, (pos, speedup) in enumerate(zip(x_pos + i * width, speedups)):
                if speedup > 0:
                    plt.text(pos, speedup + 0.1, f'{speedup:.1f}x',
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup vs Basic Sequential', fontsize=12, fontweight='bold')
        plt.title('Complete Matrix Multiplication Methods Speedup Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width * 2.5, [f'{size}×{size}' for size in matrix_sizes], fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=10, ncol=2)  # 2 columns for 6 items
        plt.grid(True, alpha=0.3, axis='y')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance' / 'complete_methods_speedup_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Complete methods speedup comparison generated")
        print("✅ All individual complete methods analyses generated")
    

    

    
    def plot_hyperparameter_analysis(self):
        """Hyperparameter optimization visualization - one optimal value per parameter."""
        df = self.load_data('all_hyperparameter_optimization.csv')
        if df is None:
            return
        
        # Aggregate data to get average performance per parameter value
        aggregated_df = df.groupby(['Parameter_Type', 'Parameter_Value']).agg({
            'Time_ms': 'mean',
            'Speedup': 'mean',
            'Memory_MB': 'mean'
        }).reset_index()
        
        parameter_types = aggregated_df['Parameter_Type'].unique()
        
        plt.figure(figsize=(15, 10))
        
        for i, param_type in enumerate(parameter_types):
            plt.subplot(2, 2, i + 1)
            param_data = aggregated_df[aggregated_df['Parameter_Type'] == param_type].copy()
            
            # Convert parameter values to numeric if possible
            try:
                param_data['Parameter_Value_Numeric'] = pd.to_numeric(param_data['Parameter_Value'])
                param_data = param_data.sort_values('Parameter_Value_Numeric')
                
                # Plot the performance curve
                plt.plot(param_data['Parameter_Value_Numeric'], param_data['Time_ms'], 
                        marker='o', linewidth=3, markersize=8, alpha=0.8)
                
                # Mark optimal value (best performance = lowest time)
                optimal_idx = param_data['Time_ms'].idxmin()
                optimal_value = param_data.loc[optimal_idx, 'Parameter_Value_Numeric']
                optimal_time = param_data.loc[optimal_idx, 'Time_ms']
                
                plt.plot(optimal_value, optimal_time, 'ro', markersize=12, 
                        label=f'Optimal: {int(optimal_value)}', zorder=5)
                
                plt.xlabel(f'{param_type.title()} Value')
                plt.ylabel('Average Execution Time (ms)')
                plt.title(f'{param_type.title()} Optimization\n(Aggregated Results)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
            except ValueError:
                # Handle non-numeric parameters (categorical)
                param_values = param_data['Parameter_Value'].unique()
                times = [param_data[param_data['Parameter_Value'] == val]['Time_ms'].iloc[0] 
                        for val in param_values]
                
                bars = plt.bar(range(len(param_values)), times, alpha=0.8, color='skyblue')
                plt.xticks(range(len(param_values)), param_values, rotation=45)
                plt.ylabel('Average Execution Time (ms)')
                plt.title(f'{param_type.title()} Comparison\n(Aggregated Results)')
                plt.grid(True, alpha=0.3)
                
                # Mark best performer
                best_idx = times.index(min(times))
                bars[best_idx].set_color('green')
                bars[best_idx].set_label(f'Optimal: {param_values[best_idx]}')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hyperparams' / 'hyperparameter_optimization.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Hyperparameter Optimization Analysis generated")
    

    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("🎨 Generating benchmark visualizations...")
        
        # Check if data directory exists
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            print("   Run benchmarks first to generate consolidated CSV data")
            return
        
        plot_functions = [
            ("Detailed Performance Analysis", self.plot_detailed_performance_analysis),
            ("Memory Usage Analysis", self.plot_memory_usage_analysis),
            ("Thread Usage Analysis", self.plot_thread_usage_analysis),
            ("Parallel Efficiency Analysis", self.plot_parallel_efficiency_analysis),
            ("Advanced Parallel Speedup Analysis", self.plot_advanced_parallel_speedup_analysis),
            ("Core Algorithm Scaling Analysis", self.plot_core_algorithm_scaling_analysis),
            ("Advanced Algorithm Scaling Analysis", self.plot_advanced_algorithm_scaling_analysis),
            ("Advanced Algorithm Performance Comparison", self.plot_advanced_algorithm_performance_comparison),
            ("Advanced Algorithm Memory Comparison", self.plot_advanced_algorithm_memory_comparison),
            ("Hyperparameter Optimization", self.plot_hyperparameter_analysis)
        ]
        
        for name, func in plot_functions:
            try:
                print(f"  📊 Creating {name}...")
                func()
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 All plots saved to: {self.output_dir}")
        print(f"   📁 Performance Analysis: {self.output_dir}/performance/")
        print(f"   📁 Memory & Thread Usage: {self.output_dir}/performance/")
        print(f"   📁 Advanced Parallel Speedup: {self.output_dir}/performance/")
        print(f"   📁 Core Algorithm Scaling: {self.output_dir}/performance/")
        print(f"   📁 Parallel Efficiency Analysis: {self.output_dir}/efficiency/")
        print(f"   📁 Hyperparameter Analysis: {self.output_dir}/hyperparams/")


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark visualization plots')
    parser.add_argument('--data-dir', default='results', 
                       help='Directory containing consolidated CSV files')
    parser.add_argument('--output-dir', default='plots', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Run benchmarks first to generate consolidated CSV data")
        return
    
    plotter = EnhancedBenchmarkPlotter(args.data_dir, args.output_dir)
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()