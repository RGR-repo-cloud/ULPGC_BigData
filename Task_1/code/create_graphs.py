#!/usr/bin/env python3
"""
Matrix Multiplication Benchmark Visualization
Creates graphs and charts from CSV benchmark results
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import glob
import os
import sys

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_log2_scale(ax, axis='y'):
    """Set up base-2 logarithmic scale with proper formatting"""
    if axis == 'y':
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.4g}'))
    elif axis == 'x':
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))

def load_latest_results():
    """Load the most recent benchmark results CSV file"""
    results_files = glob.glob("results/benchmark_results_*.csv")
    
    # Filter out summary files
    detailed_files = [f for f in results_files if not f.endswith('_summary.csv')]
    
    if not detailed_files:
        print("No detailed benchmark results found!")
        return None
    
    csv_file = max(detailed_files, key=os.path.getctime)
    print(f"Loading data from: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        return df, csv_file
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def create_performance_comparison_plot(df):
    """Create execution time comparison plot"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average execution time by language and matrix size
    avg_times = df.groupby(['Language', 'Matrix_Size'])['Execution_Time_Seconds'].mean().reset_index()
    
    # Create subplot for each language
    languages = avg_times['Language'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (lang, color) in enumerate(zip(languages, colors)):
        lang_data = avg_times[avg_times['Language'] == lang]
        plt.plot(lang_data['Matrix_Size'], lang_data['Execution_Time_Seconds'], 
                marker='o', linewidth=2.5, markersize=8, label=lang, color=color)
    
    plt.xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
    plt.ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Matrix Multiplication Performance Comparison\n(Average Execution Time by Matrix Size)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    # Base-2 logarithmic scale for better visualization of powers-of-2 matrix sizes
    setup_log2_scale(plt.gca(), 'y')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/performance_comparison.png")
    plt.show()

def create_relative_performance_plot(df):
    """Create relative performance comparison (normalized to C)"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average times
    avg_times = df.groupby(['Language', 'Matrix_Size'])['Execution_Time_Seconds'].mean().reset_index()
    
    # Find common matrix sizes between all languages
    languages = avg_times['Language'].unique()
    common_sizes = set(avg_times[avg_times['Language'] == languages[0]]['Matrix_Size'])
    for lang in languages[1:]:
        lang_sizes = set(avg_times[avg_times['Language'] == lang]['Matrix_Size'])
        common_sizes = common_sizes.intersection(lang_sizes)
    
    common_sizes = sorted(list(common_sizes))
    
    if not common_sizes:
        print("No common matrix sizes found for relative comparison")
        return
    
    # Calculate relative performance (normalized to C)
    relative_data = []
    for size in common_sizes:
        size_data = avg_times[avg_times['Matrix_Size'] == size]
        c_time = size_data[size_data['Language'] == 'C']['Execution_Time_Seconds'].iloc[0]
        
        for _, row in size_data.iterrows():
            relative_time = row['Execution_Time_Seconds'] / c_time
            relative_data.append({
                'Matrix_Size': size,
                'Language': row['Language'],
                'Relative_Performance': relative_time
            })
    
    rel_df = pd.DataFrame(relative_data)
    
    # Create the plot
    languages = ['C', 'Java', 'Python']
    colors = ['#45B7D1', '#FFA07A', '#98D8C8']
    
    x = np.arange(len(common_sizes))
    width = 0.25
    
    for i, (lang, color) in enumerate(zip(languages, colors)):
        if lang in rel_df['Language'].values:
            lang_data = rel_df[rel_df['Language'] == lang]
            values = [lang_data[lang_data['Matrix_Size'] == size]['Relative_Performance'].iloc[0] 
                     for size in common_sizes]
            plt.bar(x + i*width, values, width, label=lang, color=color, alpha=0.8)
    
    plt.xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Performance (normalized to C)', fontsize=12, fontweight='bold')
    plt.title('Relative Performance Comparison\n(C = 1.0x baseline)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x + width, [f'{size}x{size}' for size in common_sizes])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    # Linear scale for clearer visualization of performance ratios
    plt.tight_layout()
    
    plt.savefig('results/relative_performance.png', dpi=300, bbox_inches='tight')
    print("Saved: results/relative_performance.png")
    plt.show()

def create_memory_usage_plot(df):
    """Create memory usage comparison plot"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average memory usage
    avg_memory = df.groupby(['Language', 'Matrix_Size'])['Memory_Usage_MB'].mean().reset_index()
    
    languages = avg_memory['Language'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (lang, color) in enumerate(zip(languages, colors)):
        lang_data = avg_memory[avg_memory['Language'] == lang]
        plt.plot(lang_data['Matrix_Size'], lang_data['Memory_Usage_MB'], 
                marker='s', linewidth=2.5, markersize=8, label=lang, color=color)
    
    plt.xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
    plt.ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    plt.title('Memory Usage Comparison by Matrix Size', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    # Base-2 logarithmic scale for better visualization of memory scaling
    setup_log2_scale(plt.gca(), 'y')
    plt.tight_layout()
    
    plt.savefig('results/memory_usage.png', dpi=300, bbox_inches='tight')
    print("Saved: results/memory_usage.png")
    plt.show()

def create_relative_memory_plot(df):
    """Create relative memory usage comparison (normalized to C)"""
    plt.figure(figsize=(12, 8))
    
    # Calculate average memory usage
    avg_memory = df.groupby(['Language', 'Matrix_Size'])['Memory_Usage_MB'].mean().reset_index()
    
    # Find common matrix sizes between all languages
    languages = avg_memory['Language'].unique()
    common_sizes = set(avg_memory[avg_memory['Language'] == languages[0]]['Matrix_Size'])
    for lang in languages[1:]:
        lang_sizes = set(avg_memory[avg_memory['Language'] == lang]['Matrix_Size'])
        common_sizes = common_sizes.intersection(lang_sizes)
    
    common_sizes = sorted(list(common_sizes))
    
    if not common_sizes:
        print("No common matrix sizes found for relative memory comparison")
        return
    
    # Calculate relative memory usage (normalized to C)
    relative_data = []
    for size in common_sizes:
        size_data = avg_memory[avg_memory['Matrix_Size'] == size]
        c_memory = size_data[size_data['Language'] == 'C']['Memory_Usage_MB'].iloc[0]
        
        for _, row in size_data.iterrows():
            relative_memory = row['Memory_Usage_MB'] / c_memory if c_memory > 0 else 0
            relative_data.append({
                'Matrix_Size': size,
                'Language': row['Language'],
                'Relative_Memory': relative_memory
            })
    
    rel_df = pd.DataFrame(relative_data)
    
    # Create the plot
    languages = ['C', 'Java', 'Python']
    colors = ['#45B7D1', '#FFA07A', '#98D8C8']
    
    x = np.arange(len(common_sizes))
    width = 0.25
    
    for i, (lang, color) in enumerate(zip(languages, colors)):
        if lang in rel_df['Language'].values:
            lang_data = rel_df[rel_df['Language'] == lang]
            values = [lang_data[lang_data['Matrix_Size'] == size]['Relative_Memory'].iloc[0] 
                     for size in common_sizes]
            plt.bar(x + i*width, values, width, label=lang, color=color, alpha=0.8)
    
    plt.xlabel('Matrix Size (NxN)', fontsize=12, fontweight='bold')
    plt.ylabel('Relative Memory Usage (normalized to C)', fontsize=12, fontweight='bold')
    plt.title('Relative Memory Usage Comparison\n(C = 1.0x baseline)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x + width, [f'{size}x{size}' for size in common_sizes])
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig('results/relative_memory_usage.png', dpi=300, bbox_inches='tight')
    print("Saved: results/relative_memory_usage.png")
    plt.show()


def create_memory_scaling_analysis_plot(df):
    """Create memory usage scaling analysis plot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    languages = df['Language'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (lang, color) in enumerate(zip(languages, colors)):
        lang_data = df[df['Language'] == lang]
        avg_memory = lang_data.groupby('Matrix_Size')['Memory_Usage_MB'].mean()
        
        # Plot memory usage vs matrix size
        axes[i].plot(avg_memory.index, avg_memory.values, 
                    marker='s', linewidth=3, markersize=8, color=color)
        axes[i].set_title(f'{lang} Memory Usage Scaling', fontweight='bold')
        axes[i].set_xlabel('Matrix Size (NxN)', fontweight='bold')
        axes[i].set_ylabel('Memory Usage (MB)', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        # Base-2 logarithmic scale for both axes (matrix sizes and memory usage)
        setup_log2_scale(axes[i], 'y')
        setup_log2_scale(axes[i], 'x')
        
        # Add theoretical O(n²) line for memory (since we store 3 matrices)
        if len(avg_memory) > 1:
            sizes = np.array(avg_memory.index)
            first_memory = avg_memory.iloc[0]
            first_size = sizes[0]
            theoretical = first_memory * (sizes / first_size) ** 2
            axes[i].plot(sizes, theoretical, '--', alpha=0.7, color='gray', 
                        label='Theoretical O(n²)')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('results/memory_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/memory_scaling_analysis.png")
    plt.show()


def create_memory_statistical_summary_plot(df):
    """Create box plots showing memory usage statistical distribution"""
    plt.figure(figsize=(14, 10))
    
    # Create subplot for each language
    languages = df['Language'].unique()
    
    for i, lang in enumerate(languages, 1):
        plt.subplot(2, 2, i)
        lang_data = df[df['Language'] == lang]
        
        # Create box plot for each matrix size
        sizes = sorted(lang_data['Matrix_Size'].unique())
        data_for_boxplot = []
        labels = []
        
        for size in sizes:
            size_data = lang_data[lang_data['Matrix_Size'] == size]['Memory_Usage_MB']
            data_for_boxplot.append(size_data)
            labels.append(f'{size}x{size}')
        
        bp = plt.boxplot(data_for_boxplot, tick_labels=labels, patch_artist=True)
        
        # Customize colors
        colors = plt.cm.Set2(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title(f'{lang} Memory Usage Distribution', fontweight='bold')
        plt.xlabel('Matrix Size', fontweight='bold')
        plt.ylabel('Memory Usage (MB)', fontweight='bold')
        # Base-2 logarithmic scale for better visualization of memory distribution
        setup_log2_scale(plt.gca(), 'y')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/memory_statistical_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: results/memory_statistical_summary.png")
    plt.show()

def create_scaling_analysis_plot(df):
    """Create performance scaling analysis plot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    languages = df['Language'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (lang, color) in enumerate(zip(languages, colors)):
        lang_data = df[df['Language'] == lang]
        avg_times = lang_data.groupby('Matrix_Size')['Execution_Time_Seconds'].mean()
        
        # Plot execution time vs matrix size
        axes[i].plot(avg_times.index, avg_times.values, 
                    marker='o', linewidth=3, markersize=8, color=color)
        axes[i].set_title(f'{lang} Performance Scaling', fontweight='bold')
        axes[i].set_xlabel('Matrix Size (NxN)', fontweight='bold')
        axes[i].set_ylabel('Execution Time (seconds)', fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        # Base-2 logarithmic scale for both axes (matrix sizes and execution times)
        setup_log2_scale(axes[i], 'y')
        setup_log2_scale(axes[i], 'x')
        
        # Add theoretical O(n³) line for comparison
        if len(avg_times) > 1:
            sizes = np.array(avg_times.index)
            first_time = avg_times.iloc[0]
            first_size = sizes[0]
            theoretical = first_time * (sizes / first_size) ** 3
            axes[i].plot(sizes, theoretical, '--', alpha=0.7, color='gray', 
                        label='Theoretical O(n³)')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('results/scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/scaling_analysis.png")
    plt.show()

def create_statistical_summary_plot(df):
    """Create box plots showing statistical distribution"""
    plt.figure(figsize=(14, 10))
    
    # Create subplot for each language
    languages = df['Language'].unique()
    
    for i, lang in enumerate(languages, 1):
        plt.subplot(2, 2, i)
        lang_data = df[df['Language'] == lang]
        
        # Create box plot for each matrix size
        sizes = sorted(lang_data['Matrix_Size'].unique())
        data_for_boxplot = []
        labels = []
        
        for size in sizes:
            size_data = lang_data[lang_data['Matrix_Size'] == size]['Execution_Time_Seconds']
            data_for_boxplot.append(size_data)
            labels.append(f'{size}x{size}')
        
        bp = plt.boxplot(data_for_boxplot, tick_labels=labels, patch_artist=True)
        
        # Customize colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title(f'{lang} Performance Distribution', fontweight='bold')
        plt.xlabel('Matrix Size', fontweight='bold')
        plt.ylabel('Execution Time (seconds)', fontweight='bold')
        # Base-2 logarithmic scale for better visualization
        setup_log2_scale(plt.gca(), 'y')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/statistical_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: results/statistical_summary.png")
    plt.show()

def main():
    """Main function to generate all graphs"""
    print("🎨 Matrix Multiplication Benchmark Visualization")
    print("=" * 55)
    
    # Load data
    result = load_latest_results()
    if result is None:
        return
    
    df, csv_file = result
    print(f"Loaded {len(df)} benchmark runs")
    print(f"Languages: {', '.join(df['Language'].unique())}")
    print(f"Matrix sizes: {sorted(df['Matrix_Size'].unique())}")
    print()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("Generating visualizations...")
    print()
    
    # Generate all plots
    try:
        print("📊 Creating execution time visualizations...")
        create_performance_comparison_plot(df)
        create_relative_performance_plot(df)
        create_scaling_analysis_plot(df)
        create_statistical_summary_plot(df)
        
        print("💾 Creating memory usage visualizations...")
        create_memory_usage_plot(df)
        create_relative_memory_plot(df)
        create_memory_scaling_analysis_plot(df)
        create_memory_statistical_summary_plot(df)
        
        print()
        print("✅ All visualizations generated successfully!")
        print("📊 Check the results/ directory for PNG files")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

if __name__ == "__main__":
    main()