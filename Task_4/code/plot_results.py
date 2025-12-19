#!/usr/bin/env python3
"""
Plot benchmark results comparing Java and Python implementations.
Creates line plots for execution time and memory usage across matrix sizes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_combined_results():
    """Load the combined benchmark results CSV."""
    # Look for combined results file
    combined_files = glob.glob('combined_result/combined_benchmark_results*.csv')
    
    if not combined_files:
        raise FileNotFoundError("Could not find combined results file in combined_result/")
    
    # Use the most recent file if multiple exist
    combined_file = max(combined_files, key=os.path.getctime)
    
    print(f"Loading combined results from: {combined_file}")
    
    df = pd.read_csv(combined_file)
    
    return df

def plot_results(df):
    """Create separate line plots for each language and metric (4 plots total)."""
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define colors and markers for methods
    colors = {
        'basic': '#2ecc71', 
        'parallel': '#3498db', 
        'distributed_2': '#e74c3c',
        'distributed_4': '#9b59b6'
    }
    markers = {
        'basic': 'o', 
        'parallel': 's', 
        'distributed_2': '^',
        'distributed_4': 'D'
    }
    
    # Get unique matrix sizes and methods
    matrix_sizes = sorted(df['matrix_size'].unique())
    methods = ['basic', 'parallel', 'distributed']
    method_labels = {
        'basic': 'Basic', 
        'parallel': 'Parallel', 
        'distributed_2': 'Distributed (2 nodes)',
        'distributed_4': 'Distributed (4 nodes)'
    }
    languages = ['Java', 'Python']
    metrics = [
        ('time_ms_mean', 'Execution Time (ms)', 'time'),
        ('memory_kb_mean', 'Memory Usage (KB)', 'memory')
    ]
    
    output_files = []
    
    # Create 4 separate plots: 2 languages × 2 metrics
    for lang in languages:
        for metric_col, metric_label, metric_name in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get available matrix sizes for this language
            lang_data = df[df['implementation'] == lang]
            lang_matrix_sizes = sorted(lang_data['matrix_size'].unique())
            
            # Plot each method for this language
            for method in methods:
                if method == 'distributed':
                    # Plot distributed with 2 nodes
                    data_2 = df[(df['implementation'] == lang) & (df['method'] == method) & (df['cluster_nodes'] == 2)]
                    if not data_2.empty:
                        data_sorted = data_2.sort_values('matrix_size')
                        ax.plot(data_sorted['matrix_size'], data_sorted[metric_col],
                               label=method_labels['distributed_2'],
                               color=colors['distributed_2'],
                               marker=markers['distributed_2'],
                               linewidth=2.5,
                               markersize=10)
                    
                    # Plot distributed with 4 nodes
                    data_4 = df[(df['implementation'] == lang) & (df['method'] == method) & (df['cluster_nodes'] == 4)]
                    if not data_4.empty:
                        data_sorted = data_4.sort_values('matrix_size')
                        ax.plot(data_sorted['matrix_size'], data_sorted[metric_col],
                               label=method_labels['distributed_4'],
                               color=colors['distributed_4'],
                               marker=markers['distributed_4'],
                               linewidth=2.5,
                               markersize=10)
                else:
                    # Plot basic and parallel
                    data = df[(df['implementation'] == lang) & (df['method'] == method)]
                    if not data.empty:
                        data_sorted = data.sort_values('matrix_size')
                        ax.plot(data_sorted['matrix_size'], data_sorted[metric_col],
                               label=method_labels[method],
                               color=colors[method],
                               marker=markers[method],
                               linewidth=2.5,
                               markersize=10)
            
            ax.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=13, fontweight='bold')
            ax.set_title(f'{lang} - {metric_label}', fontsize=15, fontweight='bold', pad=20)
            ax.set_yscale('log', base=2)
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(lang_matrix_sizes)
            ax.set_xticklabels([f'{s}×{s}' for s in lang_matrix_sizes], rotation=45, ha='right')
            
            # Adjust layout and save
            plt.tight_layout()
            output_file = os.path.join(plots_dir, f'{lang.lower()}_{metric_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            print(f"Plot saved to: {output_file}")
            plt.close()
    
    return output_files

def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for lang in ['Java', 'Python']:
        print(f"\n{lang}:")
        lang_data = df[df['implementation'] == lang]
        for method in ['basic', 'parallel', 'distributed']:
            method_data = lang_data[lang_data['method'] == method]
            if not method_data.empty:
                print(f"  {method.capitalize()}:")
                for size in sorted(method_data['matrix_size'].unique()):
                    size_data = method_data[method_data['matrix_size'] == size].iloc[0]
                    avg_time = size_data['time_ms_mean']
                    avg_mem = size_data['memory_kb_mean']
                    print(f"    {size}×{size}: {avg_time:.2f}ms, {avg_mem:.2f} KB")

def plot_distributed_metrics(df):
    """Create plots for distributed algorithm metrics per language."""
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter for distributed method only
    dist_df = df[df['method'] == 'distributed'].copy()
    
    if dist_df.empty:
        print("No distributed data found")
        return []
    
    output_files = []
    languages = ['Java', 'Python']
    
    # Metrics to plot: (column_name, ylabel, filename_suffix)
    metrics = [
        ('network_time_ms_mean', 'Network Time (ms)', 'network_time'),
        ('data_transferred_mb_mean', 'Data Transferred (MB)', 'data_transferred'),
        ('memory_per_node_kb', 'Memory per Node (KB)', 'memory_per_node')
    ]
    
    colors_nodes = {2: '#e74c3c', 4: '#9b59b6'}
    markers_nodes = {2: '^', 4: 'D'}
    
    for lang in languages:
        lang_dist = dist_df[dist_df['implementation'] == lang]
        
        if lang_dist.empty:
            continue
        
        for metric_col, metric_label, metric_name in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot for each cluster size
            for nodes in [2, 4]:
                data = lang_dist[lang_dist['cluster_nodes'] == nodes]
                if not data.empty:
                    data_sorted = data.sort_values('matrix_size')
                    matrix_sizes = sorted(data_sorted['matrix_size'].unique())
                    
                    ax.plot(data_sorted['matrix_size'], data_sorted[metric_col],
                           label=f'{nodes} nodes',
                           color=colors_nodes[nodes],
                           marker=markers_nodes[nodes],
                           linewidth=2.5,
                           markersize=10)
            
            # Get all matrix sizes for this language's distributed data
            all_matrix_sizes = sorted(lang_dist['matrix_size'].unique())
            
            ax.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=13, fontweight='bold')
            ax.set_title(f'{lang} - Distributed {metric_label}', fontsize=15, fontweight='bold', pad=20)
            ax.set_yscale('log', base=2)
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(all_matrix_sizes)
            ax.set_xticklabels([f'{s}×{s}' for s in all_matrix_sizes], rotation=45, ha='right')
            
            # Adjust layout and save
            plt.tight_layout()
            output_file = os.path.join(plots_dir, f'{lang.lower()}_distributed_{metric_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            print(f"Plot saved to: {output_file}")
            plt.close()
    
    return output_files

def plot_network_overhead(df):
    """Create plots showing network overhead as percentage of total time for distributed method."""
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Filter for distributed method only
    dist_df = df[df['method'] == 'distributed'].copy()
    
    if dist_df.empty:
        print("No distributed data found")
        return []
    
    # Calculate network overhead percentage
    dist_df['network_overhead_pct'] = (dist_df['network_time_ms_mean'] / dist_df['time_ms_mean']) * 100
    
    output_files = []
    languages = ['Java', 'Python']
    
    for lang in languages:
        lang_dist = dist_df[dist_df['implementation'] == lang]
        
        if lang_dist.empty:
            continue
        
        # Get available matrix sizes for this language
        matrix_sizes = sorted(lang_dist['matrix_size'].unique())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separate data by cluster nodes
        data_2 = lang_dist[lang_dist['cluster_nodes'] == 2].sort_values('matrix_size')
        data_4 = lang_dist[lang_dist['cluster_nodes'] == 4].sort_values('matrix_size')
        
        # Create grouped bar chart
        x_pos = np.arange(len(matrix_sizes))
        width = 0.35
        
        if not data_2.empty:
            bars1 = ax.bar(x_pos - width/2, data_2['network_overhead_pct'],
                         width, label='2 nodes',
                         color='#e74c3c',
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1.2)
            
            # Add value labels on top of bars
            for bar, val in zip(bars1, data_2['network_overhead_pct']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        if not data_4.empty:
            bars2 = ax.bar(x_pos + width/2, data_4['network_overhead_pct'],
                         width, label='4 nodes',
                         color='#9b59b6',
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1.2)
            
            # Add value labels on top of bars
            for bar, val in zip(bars2, data_4['network_overhead_pct']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
        ax.set_ylabel('Network Overhead (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{lang} - Distributed Network Overhead', fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{s}×{s}' for s in matrix_sizes], rotation=45, ha='right')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        
        # Set y-axis to 0-100% range
        ax.set_ylim(0, 105)
        
        # Adjust layout and save
        plt.tight_layout()
        output_file = os.path.join(plots_dir, f'{lang.lower()}_network_overhead.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        output_files.append(output_file)
        print(f"Plot saved to: {output_file}")
        plt.close()
    
    return output_files

def plot_comparison_bars(df):
    """Create bar plots showing Java speedup/efficiency relative to Python for specific matrix sizes (64 and 128)."""
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    output_files = []
    
    # Matrix sizes to compare (common to both Java and Python)
    compare_sizes = [64, 128]
    
    # Metrics to plot - Java speedup/efficiency relative to Python
    metrics = [
        ('time', 'Java Speedup (vs Python)', 'speedup'),
        ('memory', 'Java Memory Efficiency (vs Python)', 'memory_efficiency')
    ]
    
    for metric_type, metric_label, metric_name in metrics:
        for size in compare_sizes:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Filter data for this matrix size
            size_data = df[df['matrix_size'] == size]
            
            if size_data.empty:
                continue
            
            metric_col = 'time_ms_mean' if metric_type == 'time' else 'memory_kb_mean'
            
            # Get data for each method
            methods = ['basic', 'parallel', 'distributed_2', 'distributed_4']
            method_labels = ['Basic', 'Parallel', 'Distributed (2)', 'Distributed (4)']
            
            # Prepare data (compute Java speedup/efficiency relative to Python)
            speedup_values = []
            labels = []
            
            for method, label in zip(methods, method_labels):
                if method == 'distributed_2':
                    java_data = size_data[(size_data['implementation'] == 'Java') & 
                                         (size_data['method'] == 'distributed') & 
                                         (size_data['cluster_nodes'] == 2)]
                    python_data = size_data[(size_data['implementation'] == 'Python') & 
                                           (size_data['method'] == 'distributed') & 
                                           (size_data['cluster_nodes'] == 2)]
                elif method == 'distributed_4':
                    java_data = size_data[(size_data['implementation'] == 'Java') & 
                                         (size_data['method'] == 'distributed') & 
                                         (size_data['cluster_nodes'] == 4)]
                    python_data = size_data[(size_data['implementation'] == 'Python') & 
                                           (size_data['method'] == 'distributed') & 
                                           (size_data['cluster_nodes'] == 4)]
                else:
                    java_data = size_data[(size_data['implementation'] == 'Java') & (size_data['method'] == method)]
                    python_data = size_data[(size_data['implementation'] == 'Python') & (size_data['method'] == method)]
                
                if not java_data.empty and not python_data.empty:
                    java_val = java_data[metric_col].iloc[0]
                    python_val = python_data[metric_col].iloc[0]
                    
                    # Calculate Java speedup/efficiency relative to Python
                    if metric_type == 'time':
                        # Speedup: Python time / Java time (how many times faster Java is)
                        speedup_values.append(python_val / java_val)
                    else:  # memory
                        # Memory efficiency: Python memory / Java memory (how much less memory Java uses)
                        speedup_values.append(python_val / java_val)
                    
                    labels.append(label)
            
            if not labels:
                continue
            
            # Create bar chart - green when Java is faster/more efficient (>1), red otherwise
            x_pos = np.arange(len(labels))
            colors = ['#2ecc71' if val >= 1.0 else '#e74c3c' for val in speedup_values]
            
            bars = ax.bar(x_pos, speedup_values,
                         color=colors,
                         alpha=0.8,
                         edgecolor='black',
                         linewidth=1.2)
            
            # Add value labels on top of bars
            for bar, val in zip(bars, speedup_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                       f'{val:.2f}x',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add horizontal line at y=1 (parity)
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
                      label='Parity (Java = Python)')
            
            ax.set_xlabel('Method', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=13, fontweight='bold')
            ax.set_title(f'Java vs Python - {metric_label} ({size}×{size})', 
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            
            # Add custom legend to explain colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', edgecolor='black', label='Java is faster/more efficient'),
                Patch(facecolor='#e74c3c', edgecolor='black', label='Python is faster/more efficient')
            ]
            ax.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9)
            
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim(bottom=0)
            
            # Adjust layout and save
            plt.tight_layout()
            output_file = os.path.join(plots_dir, f'comparison_{size}x{size}_{metric_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            print(f"Plot saved to: {output_file}")
            plt.close()
    
    return output_files

def plot_distributed_comparison(df):
    """Create bar plots comparing network time and data transferred between Java and Python for distributed methods."""
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    output_files = []
    
    # Test both 2-node and 4-node clusters
    cluster_configs = [2, 4]
    
    for cluster_nodes in cluster_configs:
        # Get distributed data for this cluster configuration
        dist_data = df[(df['method'] == 'distributed') & (df['cluster_nodes'] == cluster_nodes)].copy()
        
        if dist_data.empty:
            continue
        
        # Find common matrix sizes between Java and Python
        java_sizes = set(dist_data[dist_data['implementation'] == 'Java']['matrix_size'])
        python_sizes = set(dist_data[dist_data['implementation'] == 'Python']['matrix_size'])
        common_sizes = sorted(java_sizes.intersection(python_sizes))
        
        if not common_sizes:
            continue
        
        # Metrics to compare
        metrics = [
            ('network_time_ms_mean', 'Network Time (ms)', 'network_time'),
            ('data_transferred_mb_mean', 'Data Transferred (MB)', 'data_transferred')
        ]
        
        for metric_col, metric_label, metric_name in metrics:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            java_values = []
            python_values = []
            size_labels = []
            
            for size in common_sizes:
                java_row = dist_data[(dist_data['implementation'] == 'Java') & 
                                    (dist_data['matrix_size'] == size)]
                python_row = dist_data[(dist_data['implementation'] == 'Python') & 
                                      (dist_data['matrix_size'] == size)]
                
                if not java_row.empty and not python_row.empty:
                    java_values.append(java_row[metric_col].iloc[0])
                    python_values.append(python_row[metric_col].iloc[0])
                    size_labels.append(f'{size}×{size}')
            
            if not java_values:
                plt.close()
                continue
            
            # Create grouped bar chart
            x_pos = np.arange(len(size_labels))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, java_values, width,
                          label='Java',
                          color='#3498db',
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=1.2)
            
            bars2 = ax.bar(x_pos + width/2, python_values, width,
                          label='Python',
                          color='#e74c3c',
                          alpha=0.8,
                          edgecolor='black',
                          linewidth=1.2)
            
            # Add value labels on top of bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    label_format = f'{height:.2f}' if metric_name == 'data_transferred' else f'{height:.0f}'
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           label_format,
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Matrix Size', fontsize=13, fontweight='bold')
            ax.set_ylabel(metric_label, fontsize=13, fontweight='bold')
            ax.set_title(f'Java vs Python - {metric_label} (Distributed {cluster_nodes} Nodes)', 
                        fontsize=15, fontweight='bold', pad=20)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(size_labels, rotation=45, ha='right')
            ax.legend(loc='best', fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylim(bottom=0)
            
            # Adjust layout and save
            plt.tight_layout()
            output_file = os.path.join(plots_dir, f'comparison_distributed_{metric_name}_{cluster_nodes}nodes.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            print(f"Plot saved to: {output_file}")
            plt.close()
    
    return output_files

def main():
    """Main function."""
    print("Matrix Multiplication Benchmark Plotter")
    print("="*60)
    
    # Load data
    df = load_combined_results()
    print(f"\nLoaded {len(df)} benchmark records")
    
    # Print summary
    print_summary(df)
    
    # Create main plots
    print("\nGenerating main plots...")
    output_files = plot_results(df)
    
    # Create distributed metrics plots
    print("\nGenerating distributed metrics plots...")
    dist_files = plot_distributed_metrics(df)
    output_files.extend(dist_files)
    
    # Create network overhead plots
    print("\nGenerating network overhead plots...")
    overhead_files = plot_network_overhead(df)
    output_files.extend(overhead_files)
    
    # Create comparison bar plots
    print("\nGenerating comparison bar plots...")
    comparison_files = plot_comparison_bars(df)
    output_files.extend(comparison_files)
    
    # Create distributed comparison plots
    print("\nGenerating distributed comparison plots...")
    dist_comparison_files = plot_distributed_comparison(df)
    output_files.extend(dist_comparison_files)
    
    print(f"\nGenerated {len(output_files)} plots:")
    for f in output_files:
        print(f"  - {f}")
    print("\nDone!")

if __name__ == '__main__':
    main()
