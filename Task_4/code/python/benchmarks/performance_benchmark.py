import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'production'))

import numpy as np
import time
import tracemalloc
import psutil
import json
from typing import Dict, List
from matrix_multiplier import DistributedMatrixMultiplier, basic_multiply, parallel_multiply


class PerformanceBenchmark:
    """
    Benchmark suite for comparing different matrix multiplication approaches.
    Measures execution time, memory usage, and scalability.
    """
    
    def __init__(self):
        self.results = []
        self.multiplier = None  # Only initialize when needed for distributed
        
    def _measure_performance(self, func, *args, num_runs: int = 5, is_distributed: bool = False) -> Dict:
        """Measure execution time and memory usage of a function with multiple runs."""
        execution_times = []
        memory_used = []
        network_times = []
        data_transferred = []
        
        # Warmup runs (3 iterations)
        import gc
        print("  Warmup: ", end="", flush=True)
        for w in range(3):
            _ = func(*args)
            gc.collect()
            print(f".{w+1}", end="", flush=True)
        print(" done")
        time.sleep(0.1)
        
        for run in range(num_runs):
            # Force garbage collection for consistent memory measurements
            import gc
            gc.collect()
            
            # Start memory tracking
            tracemalloc.start()
            
            # Measure execution time
            start_time = time.time()
            result = func(*args)
            end_time = time.time()
            
            # Get peak memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            exec_time_ms = (end_time - start_time) * 1000
            mem_kb = peak / 1024
            
            execution_times.append(exec_time_ms)
            memory_used.append(mem_kb)
            
            # Print progress
            print(f"    Run {run + 1}/{num_runs} completed: {exec_time_ms:.2f}ms, {mem_kb:.2f} KB")
            
            # Collect distributed-specific metrics
            if is_distributed and hasattr(self.multiplier, 'network_time'):
                network_times.append(self.multiplier.network_time * 1000)  # Convert to ms
                data_transferred.append(self.multiplier.data_transferred_mb)
        
        metrics = {
            'execution_times': execution_times,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'memory_used_kbs': memory_used,  # Per-run memory values
            'memory_used_kb': sum(memory_used) / len(memory_used),  # Average for display
            'num_runs': num_runs,
            'result_shape': result.shape if hasattr(result, 'shape') else None
        }
        
        # Add distributed-specific metrics (both per-run and averages)
        if network_times:
            metrics['network_times'] = network_times  # Per-run network times
            metrics['data_transferred_mbs'] = data_transferred  # Per-run data transfer
            metrics['network_time_avg'] = sum(network_times) / len(network_times)
            metrics['data_transferred_mb_avg'] = sum(data_transferred) / len(data_transferred)
            metrics['computation_time_avg'] = metrics['avg_execution_time'] - metrics['network_time_avg']
            metrics['network_overhead_percent'] = (metrics['network_time_avg'] / metrics['avg_execution_time']) * 100
        
        return metrics
    
    def benchmark_matrix_sizes(self, sizes: List[int], distributed_only: bool = False, method: str = 'all') -> List[Dict]:
        """Benchmark different matrix sizes.
        
        Args:
            sizes: List of matrix sizes to test
            distributed_only: If True, only run distributed tests (deprecated, use method='distributed')
            method: Which method to run ('all', 'basic', 'parallel', 'distributed')
        """
        print("Starting matrix size benchmark...")
        
        # Only connect to Hazelcast if we need distributed tests
        distributed_available = False
        if method in ['all', 'distributed']:
            self.multiplier = DistributedMatrixMultiplier()
            if not self.multiplier.connect():
                print("Warning: Could not connect to Hazelcast. Skipping distributed tests.")
                distributed_available = False
            else:
                distributed_available = True
        
        results = []
        
        for size in sizes:
            print(f"\nTesting matrix size: {size}x{size}")
            
            # Generate test matrices
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            num_runs = 5
            
            test_result = {
                'matrix_size': size,
                'total_elements': size * size * 2,
                'memory_size_mb': (A.nbytes + B.nbytes) / 1024 / 1024,
                'num_runs': num_runs
            }
            
            if not distributed_only and method in ['all', 'basic']:
                # Test basic multiplication
                print(f"  Testing basic multiplication ({num_runs} runs)...")
                basic_perf = self._measure_performance(basic_multiply, A, B, num_runs=num_runs)
                test_result['basic'] = basic_perf
                
            if not distributed_only and method in ['all', 'parallel']:
                # Test parallel multiplication
                print(f"  Testing parallel multiplication ({num_runs} runs)...")
                parallel_perf = self._measure_performance(parallel_multiply, A, B, num_runs=num_runs)
                test_result['parallel'] = parallel_perf
            
            # Test distributed multiplication (if available and requested)
            if distributed_available and method in ['all', 'distributed']:
                print(f"  Testing distributed multiplication ({num_runs} runs)...")
                try:
                    dist_perf = self._measure_performance(self.multiplier.multiply, A, B, num_runs=num_runs, is_distributed=True)
                    dist_perf['cluster_nodes'] = self.multiplier.cluster_size
                    dist_perf['memory_per_node_kb'] = self.multiplier.memory_per_node_mb * 1024  # Convert MB to KB
                    test_result['distributed'] = dist_perf
                except Exception as e:
                    print(f"    Distributed test failed: {e}")
                    test_result['distributed'] = {'error': str(e)}
            elif method == 'distributed':
                test_result['distributed'] = {'error': 'Hazelcast not available'}
            
            results.append(test_result)
            
            # Print summary for this size
            self._print_size_summary(test_result)
        
        if distributed_available and self.multiplier:
            self.multiplier.disconnect()
        
        return results
    
    def _print_size_summary(self, result: Dict):
        """Print summary for a single matrix size test."""
        size = result['matrix_size']
        print(f"\n  Results for {size}x{size} matrices (average of {result.get('num_runs', 5)} runs):")
        
        if 'basic' in result:
            avg_time = result['basic'].get('avg_execution_time', result['basic'].get('execution_time', 0))
            mem = result['basic'].get('memory_used_kb', 0)
            print(f"    Basic:       {avg_time:.2f}ms (min: {result['basic'].get('min_execution_time', avg_time):.2f}ms, max: {result['basic'].get('max_execution_time', avg_time):.2f}ms, Mem: {mem:.2f} KB)")
        if 'parallel' in result:
            avg_time = result['parallel'].get('avg_execution_time', result['parallel'].get('execution_time', 0))
            mem = result['parallel'].get('memory_used_kb', 0)
            print(f"    Parallel:    {avg_time:.2f}ms (min: {result['parallel'].get('min_execution_time', avg_time):.2f}ms, max: {result['parallel'].get('max_execution_time', avg_time):.2f}ms, Mem: {mem:.2f} KB)")
        if 'distributed' in result:
            if 'avg_execution_time' in result['distributed'] or 'execution_time' in result['distributed']:
                avg_time = result['distributed'].get('avg_execution_time', result['distributed'].get('execution_time', 0))
                network_time = result['distributed'].get('network_time_avg', 0)
                network_pct = result['distributed'].get('network_overhead_percent', 0)
                data_mb = result['distributed'].get('data_transferred_mb_avg', 0)
                nodes = result['distributed'].get('cluster_nodes', 1)
                mem_kb = result['distributed'].get('memory_used_kb', 0)
                print(f"    Distributed: {avg_time:.2f}ms (min: {result['distributed'].get('min_execution_time', avg_time):.2f}ms, max: {result['distributed'].get('max_execution_time', avg_time):.2f}ms)")
                print(f"                 Network: {network_time:.2f}ms ({network_pct:.1f}% overhead), Data: {data_mb:.2f} MB transferred")
                print(f"                 Cluster: {nodes} node(s), Memory: {mem_kb:.2f} KB)")
            else:
                print(f"    Distributed: Failed ({result['distributed']['error']})")
    
    def run_scalability_test(self, sizes=None, distributed_only: bool = False, method: str = 'all') -> Dict:
        """Run comprehensive scalability test."""
        if sizes is None:
            sizes = [64, 128, 256]
        
        print("=" * 60)
        print("SCALABILITY BENCHMARK")
        print("=" * 60)
        
        # Test different matrix sizes
        results = self.benchmark_matrix_sizes(sizes, distributed_only=distributed_only, method=method)
        
        # Calculate scalability metrics
        scalability_analysis = self._analyze_scalability(results)
        
        return {
            'benchmark_results': results,
            'scalability_analysis': scalability_analysis,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self._get_system_info()
        }
    
    def _analyze_scalability(self, results: List[Dict]) -> Dict:
        """Analyze scalability from benchmark results."""
        analysis = {}
        
        if len(results) < 2:
            return analysis
        
        # Analyze each method
        for method in ['basic', 'parallel', 'distributed']:
            method_times = []
            method_sizes = []
            
            for result in results:
                if method in result:
                    # Use avg_execution_time if available, otherwise execution_time
                    exec_time = result[method].get('avg_execution_time', result[method].get('execution_time'))
                    if exec_time is not None:
                        method_times.append(exec_time)
                        method_sizes.append(result['matrix_size'])
            
            if len(method_times) >= 2:
                # Calculate speedup relative to smallest size
                base_time = method_times[0]
                speedups = [base_time / t if t > 0 else 0 for t in method_times]
                
                # Calculate efficiency (theoretical vs actual)
                size_ratios = [s / method_sizes[0] for s in method_sizes]
                theoretical_ratios = [r ** 3 for r in size_ratios]  # O(n³) complexity
                
                analysis[method] = {
                    'execution_times': method_times,
                    'matrix_sizes': method_sizes,
                    'speedups': speedups,
                    'size_ratios': size_ratios,
                    'theoretical_complexity_ratios': theoretical_ratios
                }
        
        return analysis
    
    def _get_system_info(self) -> Dict:
        """Get system information for benchmark context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'total_memory_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'python_version': sys.version.split()[0]
        }
    
    def save_results(self, results: Dict, filename: str = None, method_suffix: str = None):
        """Save benchmark results to CSV file (matching Java format)."""
        if filename is None:
            timestamp = int(time.time() * 1000)  # milliseconds since epoch
            
            # Get cluster size from distributed results if available
            cluster_size = 0
            if method_suffix == 'distributed' and results.get('benchmark_results'):
                for result in results['benchmark_results']:
                    if 'distributed' in result and 'cluster_nodes' in result['distributed']:
                        cluster_size = result['distributed']['cluster_nodes']
                        break
            
            if method_suffix:
                if method_suffix == 'distributed' and cluster_size > 0:
                    filename = f'benchmark_results_python_{method_suffix}_{cluster_size}nodes_{timestamp}.csv'
                else:
                    filename = f'benchmark_results_python_{method_suffix}_{timestamp}.csv'
            else:
                filename = f'benchmark_results_python_{timestamp}.csv'
        
        # Save to results directory
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        # Write CSV in same format as Java
        with open(filepath, 'w') as f:
            f.write('matrix_size,method,run,time_ms,network_time_ms,data_transferred_kb,memory_kb,cluster_nodes,memory_per_node_kb\n')
            
            for result in results.get('benchmark_results', []):
                size = result['matrix_size']
                
                # Write basic results
                if 'basic' in result:
                    basic = result['basic']
                    memory_kbs = basic.get('memory_used_kbs', [])
                    for i, time_ms in enumerate(basic['execution_times'], 1):
                        mem_kb = memory_kbs[i-1] if i-1 < len(memory_kbs) else basic.get('memory_used_kb', 0)
                        f.write(f'{size},basic,{i},{time_ms:.2f},0,0,{mem_kb:.2f},0,0\n')
                
                # Write parallel results
                if 'parallel' in result:
                    parallel = result['parallel']
                    memory_kbs = parallel.get('memory_used_kbs', [])
                    for i, time_ms in enumerate(parallel['execution_times'], 1):
                        mem_kb = memory_kbs[i-1] if i-1 < len(memory_kbs) else parallel.get('memory_used_kb', 0)
                        f.write(f'{size},parallel,{i},{time_ms:.2f},0,0,{mem_kb:.2f},0,0\n')
                
                # Write distributed results
                if 'distributed' in result and 'avg_execution_time' in result['distributed']:
                    dist = result['distributed']
                    cluster_nodes = dist.get('cluster_nodes', 1)
                    memory_per_node_kb = dist.get('memory_per_node_kb', 0)
                    network_times = dist.get('network_times', [])
                    data_mbs = dist.get('data_transferred_mbs', [])
                    memory_kbs = dist.get('memory_used_kbs', [])
                    
                    for i, time_ms in enumerate(dist['execution_times'], 1):
                        # Use per-run values if available, otherwise use averages
                        network_ms = network_times[i-1] if i-1 < len(network_times) else dist.get('network_time_avg', 0)
                        data_mb = data_mbs[i-1] if i-1 < len(data_mbs) else dist.get('data_transferred_mb_avg', 0)
                        data_kb = data_mb * 1024  # Convert MB to KB
                        mem_kb = memory_kbs[i-1] if i-1 < len(memory_kbs) else dist.get('memory_used_kb', 0)
                        f.write(f'{size},distributed,{i},{time_ms:.2f},{network_ms:.2f},{data_kb:.2f},{mem_kb:.2f},{cluster_nodes},{memory_per_node_kb:.2f}\n')
        
        print(f"\nResults saved to: {filepath}")
        return filepath


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Matrix Multiplication Benchmark')
    parser.add_argument('--sizes', type=str, default='16,32,64,128',
                        help='Comma-separated matrix sizes to test (e.g., 16,32,64,128)')
    parser.add_argument('--method', type=str, choices=['basic', 'parallel', 'distributed', 'all'], 
                        default='all',
                        help='Which method to benchmark (basic, parallel, distributed, or all)')
    args = parser.parse_args()
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    # Determine which methods to run
    if args.method == 'all':
        distributed_only = False
        method_suffix = None
    elif args.method == 'distributed':
        distributed_only = True
        method_suffix = 'distributed'
    elif args.method in ['basic', 'parallel']:
        # For basic/parallel only, we'll still need to run both but filter in save
        distributed_only = False
        method_suffix = args.method
    else:
        distributed_only = False
        method_suffix = None
    
    benchmark = PerformanceBenchmark()
    
    print("Matrix Multiplication Benchmark")
    print("=" * 50)
    print(f"Mode: {args.method.capitalize()} method{'s' if args.method == 'all' else ''}")
    
    # Run scalability test with custom sizes
    results = benchmark.run_scalability_test(sizes, distributed_only=(args.method == 'distributed'), method=args.method)
    
    # Save results
    filepath = benchmark.save_results(results, method_suffix=method_suffix)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if 'scalability_analysis' in results:
        analysis = results['scalability_analysis']
        
        print("\nExecution time trends:")
        for method, data in analysis.items():
            if 'execution_times' in data:
                times = data['execution_times']
                sizes = data['matrix_sizes']
                print(f"  {method.capitalize()}:")
                for size, exec_time in zip(sizes, times):
                    print(f"    {size}x{size}: {exec_time:.2f}ms")
    
    print(f"\nDetailed results saved to: {filepath}")
    print("\nRecommendations:")
    print("- For small matrices (< 200x200): Use parallel (NumPy)")
    print("- For large matrices (> 1000x1000): Use distributed (Hazelcast)")
    print("- Network overhead is significant for small matrices")
    print("- Distributed approach scales better with matrix size")


if __name__ == "__main__":
    main()