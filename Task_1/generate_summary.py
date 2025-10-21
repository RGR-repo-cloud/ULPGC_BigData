#!/usr/bin/env python3
"""
Generates summary statistics from detailed CSV benchmark results
"""

import pandas as pd
import sys
import glob
import os

def generate_summary_from_csv(detailed_csv_file):
    """Generate summary statistics from detailed benchmark CSV"""
    
    try:
        # Read the detailed CSV
        df = pd.read_csv(detailed_csv_file)
        
        # Calculate summary statistics grouped by Language and Matrix_Size
        summary = df.groupby(['Language', 'Matrix_Size']).agg({
            'Execution_Time_Seconds': ['mean', 'min', 'max', 'std'],
            'Memory_Usage_MB': ['mean', 'max']
        }).round(6)
        
        # Flatten the multi-level column names
        summary.columns = ['Avg_Time_Seconds', 'Min_Time_Seconds', 'Max_Time_Seconds', 
                          'Std_Dev_Seconds', 'Avg_Memory_MB', 'Peak_Memory_MB']
        
        # Reset index to get Language and Matrix_Size as regular columns
        summary_df = summary.reset_index()
        
        # Create summary file name
        summary_file = detailed_csv_file.replace('.csv', '_summary.csv')
        
        # Write summary CSV
        summary_df.to_csv(summary_file, index=False)
        
        print(f"✅ Generated summary: {summary_file}")
        print(f"📊 Summary contains {len(summary_df)} entries")
        
        # Display summary for verification
        print("\n=== SUMMARY STATISTICS ===")
        for _, row in summary_df.iterrows():
            lang = row['Language']
            size = int(row['Matrix_Size'])
            avg_time = row['Avg_Time_Seconds']
            std_time = row['Std_Dev_Seconds']
            avg_mem = row['Avg_Memory_MB']
            peak_mem = row['Peak_Memory_MB']
            
            print(f"{lang:8} {size:4}x{size:<4}: "
                  f"Time={avg_time:.6f}s (±{std_time:.6f}), "
                  f"Memory={avg_mem:.2f}MB (peak {peak_mem:.2f}MB)")
        
        return summary_file
        
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return None

def find_latest_detailed_csv():
    """Find the most recent detailed benchmark CSV file"""
    results_files = glob.glob("results/benchmark_results_*.csv")
    
    # Filter out summary files
    detailed_files = [f for f in results_files if not f.endswith('_summary.csv')]
    
    if not detailed_files:
        return None
    
    # Get the most recent file
    return max(detailed_files, key=os.path.getctime)

def generate_relative_performance_summary(summary_file):
    """Generate additional relative performance analysis"""
    try:
        df = pd.read_csv(summary_file)
        
        print("\n=== RELATIVE PERFORMANCE ANALYSIS ===")
        
        # Find common matrix sizes across all languages
        languages = df['Language'].unique()
        common_sizes = set(df[df['Language'] == languages[0]]['Matrix_Size'])
        
        for lang in languages[1:]:
            lang_sizes = set(df[df['Language'] == lang]['Matrix_Size'])
            common_sizes = common_sizes.intersection(lang_sizes)
        
        common_sizes = sorted(list(common_sizes))
        
        if not common_sizes:
            print("No common matrix sizes found for comparison")
            return
        
        # Performance comparison for common sizes
        for size in common_sizes:
            size_data = df[df['Matrix_Size'] == size]
            
            # Get baseline (C if available, otherwise first language)
            if 'C' in size_data['Language'].values:
                baseline_time = size_data[size_data['Language'] == 'C']['Avg_Time_Seconds'].iloc[0]
                baseline_lang = 'C'
            else:
                baseline_time = size_data['Avg_Time_Seconds'].min()
                baseline_lang = size_data[size_data['Avg_Time_Seconds'] == baseline_time]['Language'].iloc[0]
            
            print(f"\nMatrix {size}x{size} (baseline: {baseline_lang}):")
            
            for _, row in size_data.iterrows():
                lang = row['Language']
                time = row['Avg_Time_Seconds']
                ratio = time / baseline_time if baseline_time > 0 else 0
                
                if lang == baseline_lang:
                    print(f"  {lang:8}: {time:.6f}s (baseline)")
                else:
                    print(f"  {lang:8}: {time:.6f}s ({ratio:.1f}x slower)")
    
    except Exception as e:
        print(f"Warning: Could not generate relative performance analysis: {e}")

def main():
    print("🔄 Universal Benchmark Summary Generator")
    print("=" * 45)
    
    # Determine which CSV to process
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            return
        if csv_file.endswith('_summary.csv'):
            print("❌ Please provide the detailed CSV file, not the summary file")
            return
    else:
        csv_file = find_latest_detailed_csv()
        if not csv_file:
            print("❌ No benchmark results found!")
            print("Run ./run_all_benchmarks.sh first to generate results.")
            return
        print(f"📁 Using latest results: {csv_file}")
    
    # Generate summary
    summary_file = generate_summary_from_csv(csv_file)
    
    if summary_file:
        # Generate additional analysis
        generate_relative_performance_summary(summary_file)
        
        print(f"\n✅ Complete! Files generated:")
        print(f"   📊 Detailed: {csv_file}")
        print(f"   📈 Summary:  {summary_file}")
        print(f"\n💡 Tip: Use 'python3 create_graphs.py' for visualizations")
    
if __name__ == "__main__":
    main()