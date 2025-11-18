#!/usr/bin/env python3
"""
Matrix Multiplication Performance Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def plot_dense_execution_times_all_sizes():
    """Plot execution times for all matrix sizes in separate plots"""
    sizes = [256, 512, 1024, 2048]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for size in sizes:
        try:
            df = pd.read_csv(f'results/dense_{size}x{size}.csv')
            algorithms = df['Algorithm'].tolist()
            times = df['Avg_Time_s'].tolist()
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(algorithms, times, color=colors[:len(algorithms)], alpha=0.8)
            plt.title(f'Dense Matrix Multiplication Performance - {size}×{size}', fontsize=16, fontweight='bold')
            plt.ylabel('Execution Time (seconds)')
            plt.xlabel('Algorithm')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            # Add value labels directly on top of bars
            for bar, time in zip(bars, times):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{time:.3f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'plots/dense_execution_times_{size}x{size}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Dense execution times {size}×{size} plot saved")
            
        except FileNotFoundError:
            print(f"⚠️  No data file for {size}×{size} matrix")
        except Exception as e:
            print(f"⚠️  Could not create {size}×{size} execution times plot: {e}")

def plot_dense_speedup_all_sizes():
    """Plot speedup for all matrix sizes in separate plots"""
    sizes = [256, 512, 1024, 2048]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for size in sizes:
        try:
            df = pd.read_csv(f'results/dense_{size}x{size}.csv')
            algorithms = df['Algorithm'].tolist()
            times = df['Avg_Time_s'].tolist()
            baseline = times[0]  # Basic algorithm
            speedups = [baseline / t for t in times]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(algorithms, speedups, color=colors[:len(algorithms)], alpha=0.8)
            plt.title(f'Speedup over Basic Algorithm - {size}×{size}', fontsize=16, fontweight='bold')
            plt.ylabel('Speedup (×)')
            plt.xlabel('Algorithm')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(speedups) * 1.2)  # Extra space for labels
            
            # Add value labels directly on top of bars
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{speedup:.1f}×', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'plots/dense_speedup_{size}x{size}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Dense speedup {size}×{size} plot saved")
            
        except FileNotFoundError:
            print(f"⚠️  No data file for {size}×{size} matrix")
        except Exception as e:
            print(f"⚠️  Could not create {size}×{size} speedup plot: {e}")

def plot_dense_scaling():
    """Plot algorithm scaling with matrix size"""
    try:
        sizes = [256, 512, 1024, 2048]
        algorithms = []
        times = {256: [], 512: [], 1024: [], 2048: []}
        
        # Read data from each size file
        for size in sizes:
            try:
                df = pd.read_csv(f'results/dense_{size}x{size}.csv')
                if size == 256:  # Get algorithm names
                    algorithms = df['Algorithm'].tolist()
                
                for _, row in df.iterrows():
                    times[size].append(row['Avg_Time_s'])
            except FileNotFoundError:
                continue
        
        if not algorithms:
            print("⚠️  No dense benchmark data found")
            return
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, alg in enumerate(algorithms):
            alg_times = [times[size][i] for size in sizes if times[size]]
            available_sizes = [size for size in sizes if times[size]]
            if alg_times:
                plt.plot(available_sizes, alg_times, 'o-', label=alg, 
                        linewidth=3, markersize=8, color=colors[i % len(colors)])
        
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Algorithm Scaling with Matrix Size')
        plt.yscale('log', base=2)
        plt.xscale('log', base=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/dense_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Dense scaling plot saved")
        
    except Exception as e:
        print(f"⚠️  Could not create scaling plot: {e}")

def plot_sparse_speedup():
    """Plot sparse matrix speedup vs sparsity level"""
    try:
        df = pd.read_csv('results/sparse_results.csv')
        sparse_data = df[df['Format'] == 'Sparse']
        
        sparsity = sparse_data['Sparsity_Percent'].values
        speedups = sparse_data['Speedup'].values
        
        plt.figure(figsize=(10, 6))
        plt.plot(sparsity, speedups, 'o-', linewidth=4, markersize=10, color='#2ca02c')
        
        plt.xlabel('Sparsity Level (%)')
        plt.ylabel('Speedup (×)')
        plt.title('Sparse Matrix Speedup vs Sparsity Level')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(speedups) * 1.1)
        
        # Add value labels
        for x, y in zip(sparsity, speedups):
            plt.text(x, y + max(speedups)*0.02, f'{y:.1f}×', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/sparse_speedup.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sparse speedup plot saved")
        
    except FileNotFoundError:
        print("⚠️  sparse_results.csv not found")

def plot_sparse_memory_savings():
    """Plot memory savings vs sparsity level"""
    try:
        df = pd.read_csv('results/sparse_results.csv')
        sparse_data = df[df['Format'] == 'Sparse']
        
        sparsity = sparse_data['Sparsity_Percent'].values
        memory_savings = sparse_data['Memory_Savings_Percent'].values
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in memory_savings]
        bars = plt.bar(sparsity, memory_savings, color=colors, alpha=0.7)
        
        plt.xlabel('Sparsity Level (%)')
        plt.ylabel('Memory Savings (%)')
        plt.title('Memory Savings vs Sparsity Level')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, savings in zip(bars, memory_savings):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., 
                    height + (2 if height >= 0 else -5),
                    f'{savings:.1f}%', ha='center', 
                    va='bottom' if height >= 0 else 'top',
                    fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/sparse_memory_savings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sparse memory savings plot saved")
        
    except FileNotFoundError:
        print("⚠️  sparse_results.csv not found")

def plot_sparse_execution_comparison():
    """Plot execution time comparison: dense vs sparse"""
    try:
        df = pd.read_csv('results/sparse_results.csv')
        dense_data = df[df['Format'] == 'Dense']
        sparse_data = df[df['Format'] == 'Sparse']
        
        sparsity = sparse_data['Sparsity_Percent'].values
        dense_times = dense_data['Avg_Time_s'].values
        sparse_times = sparse_data['Avg_Time_s'].values
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(sparsity))
        width = 0.35
        
        plt.bar(x - width/2, dense_times, width, label='Dense', alpha=0.8, color='#1f77b4')
        plt.bar(x + width/2, sparse_times, width, label='Sparse', alpha=0.8, color='#ff7f0e')
        
        plt.xlabel('Sparsity Level (%)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time: Dense vs Sparse')
        plt.xticks(x, [f'{int(s)}%' for s in sparsity])
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/sparse_execution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sparse execution comparison plot saved")
        
    except FileNotFoundError:
        print("⚠️  sparse_results.csv not found")

def plot_maximum_sizes():
    """Plot maximum matrix size comparison"""
    try:
        df = pd.read_csv('results/max_matrix_sizes.csv', skiprows=6)
        
        algorithms = df['Algorithm'].values
        max_sizes = df['Max_Matrix_Size'].values
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(algorithms, max_sizes, color=colors, alpha=0.8)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Maximum Matrix Size (N×N)')
        plt.title('Maximum Efficient Matrix Size (1s timeout)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars, max_sizes):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/maximum_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Maximum sizes plot saved")
        
    except FileNotFoundError:
        print("⚠️  max_matrix_sizes.csv not found")

def plot_memory_requirements():
    """Plot memory requirements at maximum sizes"""
    try:
        df = pd.read_csv('results/max_matrix_sizes.csv', skiprows=6)
        
        algorithms = df['Algorithm'].values
        memory_mb = df['Memory_Requirement_MB'].values
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(algorithms, memory_mb, color=colors, alpha=0.8)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Memory Requirement (MB)')
        plt.title('Memory Requirements at Maximum Size')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mem in zip(bars, memory_mb):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/memory_requirements.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Memory requirements plot saved")
        
    except FileNotFoundError:
        print("⚠️  max_matrix_sizes.csv not found")

def plot_dense_memory_usage_all_sizes():
    """Plot memory usage comparison for each matrix size individually"""
    sizes = [256, 512, 1024, 2048]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for size in sizes:
        try:
            df = pd.read_csv(f'results/dense_{size}x{size}.csv')
            algorithms = df['Algorithm'].tolist()
            memory_kb = df['Avg_Memory_KB'].tolist()
            memory_mb = [mem / 1024.0 for mem in memory_kb]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(algorithms, memory_mb, color=colors[:len(algorithms)], alpha=0.8)
            plt.title(f'Memory Usage Comparison - {size}×{size}', fontsize=16, fontweight='bold')
            plt.ylabel('Memory Usage (MB)')
            plt.xlabel('Algorithm')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(memory_mb) * 1.2)  # Extra space for labels
            
            # Add value labels directly on top of bars
            for bar, mem in zip(bars, memory_mb):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem:.1f}MB', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'plots/dense_memory_usage_{size}x{size}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Dense memory usage {size}×{size} plot saved")
            
        except FileNotFoundError:
            print(f"⚠️  No data file for {size}×{size} matrix")
        except Exception as e:
            print(f"⚠️  Could not create {size}×{size} memory usage plot: {e}")

def plot_sparse_memory_comparison():
    """Plot memory usage comparison between dense and sparse formats"""
    try:
        df = pd.read_csv('results/sparse_results.csv')
        dense_data = df[df['Format'] == 'Dense']
        sparse_data = df[df['Format'] == 'Sparse']
        
        sparsity = sparse_data['Sparsity_Percent'].values
        dense_memory = dense_data['Avg_Memory_KB'].values / 1024.0  # Convert to MB
        sparse_memory = sparse_data['Avg_Memory_KB'].values / 1024.0
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(sparsity))
        width = 0.35
        
        plt.bar(x - width/2, dense_memory, width, label='Dense Format', alpha=0.8, color='#1f77b4')
        plt.bar(x + width/2, sparse_memory, width, label='Sparse (CSR) Format', alpha=0.8, color='#ff7f0e')
        
        plt.xlabel('Sparsity Level (%)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage: Dense vs Sparse Format (256×256)')
        plt.xticks(x, [f'{int(s)}%' for s in sparsity])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (dense_mem, sparse_mem) in enumerate(zip(dense_memory, sparse_memory)):
            plt.text(i - width/2, dense_mem + 0.02, f'{dense_mem:.1f}MB', 
                    ha='center', va='bottom', fontsize=10)
            plt.text(i + width/2, sparse_mem + 0.02, f'{sparse_mem:.1f}MB', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('plots/sparse_memory_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Sparse memory comparison plot saved")
        
    except FileNotFoundError:
        print("⚠️  sparse_results.csv not found")

def plot_memory_scaling():
    """Plot memory usage scaling with matrix size"""
    try:
        sizes = [256, 512, 1024, 2048]
        algorithms = []
        memory_usage = {256: [], 512: [], 1024: [], 2048: []}
        
        # Read data from each size file
        for size in sizes:
            try:
                df = pd.read_csv(f'results/dense_{size}x{size}.csv')
                if size == 256:  # Get algorithm names
                    algorithms = df['Algorithm'].tolist()
                
                for _, row in df.iterrows():
                    memory_usage[size].append(row['Avg_Memory_KB'] / 1024.0)  # Convert to MB
            except FileNotFoundError:
                continue
        
        if not algorithms:
            print("⚠️  No dense benchmark data found for memory scaling")
            return
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, alg in enumerate(algorithms):
            alg_memory = [memory_usage[size][i] for size in sizes if memory_usage[size]]
            available_sizes = [size for size in sizes if memory_usage[size]]
            if alg_memory:
                plt.plot(available_sizes, alg_memory, 'o-', label=alg, 
                        linewidth=3, markersize=8, color=colors[i % len(colors)])
        
        plt.xlabel('Matrix Size (N×N)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage Scaling with Matrix Size')
        plt.yscale('log', base=2)
        plt.xscale('log', base=2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/memory_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Memory scaling plot saved")
        
    except Exception as e:
        print(f"⚠️  Could not create memory scaling plot: {e}")

def plot_memory_efficiency_all_sizes():
    """Plot memory efficiency (performance per MB) for each matrix size individually"""
    sizes = [256, 512, 1024, 2048]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for size in sizes:
        try:
            df = pd.read_csv(f'results/dense_{size}x{size}.csv')
            algorithms = df['Algorithm'].tolist()
            times = df['Avg_Time_s'].tolist()
            memory_mb = df['Avg_Memory_KB'].tolist()
            memory_mb = [mem / 1024.0 for mem in memory_mb]
            
            # Calculate GFLOPS per MB (efficiency metric)
            flops = 2 * size**3  # Matrix multiplication FLOPS
            efficiency = []
            for time, mem in zip(times, memory_mb):
                gflops = flops / (time * 1e9)
                eff = gflops / mem if mem > 0 else 0
                efficiency.append(eff)
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(algorithms, efficiency, color=colors[:len(algorithms)], alpha=0.8)
            plt.title(f'Memory Efficiency: Performance per MB - {size}×{size}', fontsize=16, fontweight='bold')
            plt.ylabel('Efficiency (GFLOPS/MB)')
            plt.xlabel('Algorithm')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, max(efficiency) * 1.2)  # Extra space for labels
            
            # Add value labels directly on top of bars
            for bar, eff in zip(bars, efficiency):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{eff:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'plots/memory_efficiency_{size}x{size}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Memory efficiency {size}×{size} plot saved")
            
        except FileNotFoundError:
            print(f"⚠️  No data file for {size}×{size} matrix")
        except Exception as e:
            print(f"⚠️  Could not create {size}×{size} memory efficiency plot: {e}")

def create_individual_plots():
    """Create all individual plots"""
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    print("🎨 Generating Individual Performance Plots...")
    print("=" * 50)
    
    # Dense algorithm plots (consolidated)
    plot_dense_execution_times_all_sizes()
    plot_dense_speedup_all_sizes()
    plot_dense_scaling()
    
    # Sparse matrix plots
    plot_sparse_speedup()
    plot_sparse_memory_savings()
    plot_sparse_execution_comparison()
    
    # Maximum size plots
    plot_maximum_sizes()
    plot_memory_requirements()
    
    # Memory usage plots (consolidated)
    plot_dense_memory_usage_all_sizes()
    plot_sparse_memory_comparison()
    plot_memory_scaling()
    plot_memory_efficiency_all_sizes()
    
    print("\n" + "=" * 50)
    print("📊 Individual Plot Generation Complete!")
    print("\nGenerated files:")
    print("  📈 plots/dense_execution_times_all_sizes.png")
    print("  ⚡ plots/dense_speedup_all_sizes.png")
    print("  📊 plots/dense_scaling.png")
    print("  🚀 plots/sparse_speedup.png")
    print("  💾 plots/sparse_memory_savings.png")
    print("  ⏱️  plots/sparse_execution_comparison.png")
    print("  📏 plots/maximum_sizes.png")
    print("  🧠 plots/memory_requirements.png")
    print("  🎯 plots/dense_memory_usage_all_sizes.png")
    print("  🔄 plots/sparse_memory_comparison.png")
    print("  📈 plots/memory_scaling.png")
    print("  ⚡ plots/memory_efficiency_all_sizes.png")
    print("\n💡 Each plot focuses on one specific performance aspect!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_name = sys.argv[1]
        if plot_name == "dense_times":
            plot_dense_execution_times_all_sizes()
        elif plot_name == "dense_speedup":
            plot_dense_speedup_all_sizes()
        elif plot_name == "dense_scaling":
            plot_dense_scaling()
        elif plot_name == "sparse_speedup":
            plot_sparse_speedup()
        elif plot_name == "sparse_memory":
            plot_sparse_memory_savings()
        elif plot_name == "sparse_comparison":
            plot_sparse_execution_comparison()
        elif plot_name == "max_sizes":
            plot_maximum_sizes()
        elif plot_name == "memory_req":
            plot_memory_requirements()
        elif plot_name == "dense_memory":
            plot_dense_memory_usage_all_sizes()
        elif plot_name == "sparse_memory_comp":
            plot_sparse_memory_comparison()
        elif plot_name == "memory_scaling":
            plot_memory_scaling()
        elif plot_name == "memory_efficiency":
            plot_memory_efficiency_all_sizes()
        else:
            print("Available plots:")
            print("  Dense: dense_times, dense_speedup, dense_scaling, dense_memory")
            print("  Sparse: sparse_speedup, sparse_memory, sparse_comparison, sparse_memory_comp")
            print("  Max Size: max_sizes, memory_req")
            print("  Memory: memory_scaling, memory_efficiency")
    else:
        create_individual_plots()