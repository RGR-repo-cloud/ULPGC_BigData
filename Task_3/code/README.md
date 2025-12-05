# Task 3: Parallel Matrix Multiplication

## Project Structure

### Production Code (`src/main/`)
- `MatrixMultiplier.java` - Interface for different multiplication strategies
- `BasicMatrixMultiplier.java` - Sequential matrix multiplication (baseline)
- `ParallelMatrixMultiplier.java` - Multi-threaded matrix multiplication
- `AdvancedParallelMatrixMultiplier.java` - Advanced parallel implementation with semaphore control
- `ForkJoinMatrixMultiplier.java` - Fork-Join framework implementation
- `VectorizedMatrixMultiplier.java` - SIMD-inspired vectorized approach
- `MatrixUtils.java` - Utility functions for matrix operations

### Benchmarking Code (`src/benchmark/`)
- `PerformanceBenchmark.java` - Main benchmarking class with comprehensive analysis
- `BenchmarkResult.java` - Data structure for storing benchmark results
- `BenchmarkRunner.java` - Utility for running and analyzing benchmarks
- `CSVExporter.java` - Advanced CSV export with duplicate handling

### Visualization (`plotting.py`)
- Comprehensive matplotlib/seaborn plotting system
- Performance analysis, memory usage, efficiency plots
- Advanced algorithm scaling analysis
- Clean plot generation with proper font handling

## How to Run

1. Build the project:
```bash
./build.sh
```

2. Run the comprehensive benchmark:
```bash
cd build && java benchmark.PerformanceBenchmark
```

3. Generate visualizations:
```bash
python3 plotting.py
```

## Requirements Covered

- ✅ Multiple parallel implementations (6 different algorithms)
- ✅ Large matrix testing (128×128 to 1024×1024)
- ✅ Vectorized approach implementation
- ✅ Advanced parallel patterns (Fork-Join, Streams, Semaphore control)
- ✅ Comprehensive performance analysis
- ✅ Memory usage tracking
- ✅ Speedup and efficiency metrics
- ✅ Professional visualization system
- ✅ Clean CSV data export

## Implemented Algorithms

1. **Basic Sequential** - Baseline triple-nested loop
2. **Vectorized (Block-based)** - Cache-optimized block multiplication
3. **Parallel (8 threads)** - Standard thread pool parallelization
4. **Advanced Parallel** - Enhanced parallel with memory optimization
5. **Advanced + Semaphore** - Semaphore-controlled resource management
6. **Parallel Streams** - Java 8 parallel streams implementation
7. **Fork-Join** - Divide-and-conquer recursive parallelism

## Key Features

- **Multiple Parallel Strategies**: 6 different parallel implementations
- **Real Memory Measurement**: Runtime.getRuntime() heap usage tracking
- **Thread Optimization**: Capped at physical core count (8 threads)
- **Advanced CSV Export**: Proper parsing of quoted algorithm names
- **Comprehensive Visualization**: matplotlib/seaborn with scaling analysis
- **Duplicate Prevention**: Clean data generation without accumulation
- **Font Optimization**: Unicode character handling for clean plots

## Matrix Sizes Tested

- 128×128 (16,384 elements)
- 256×256 (65,536 elements) 
- 512×512 (262,144 elements)
- 1024×1024 (1,048,576 elements)

## Generated Plots

### Performance Analysis
- Individual performance analysis per matrix size
- Speedup comparison across all sizes
- Memory usage analysis
- Thread usage analysis

### Efficiency Analysis
- Parallel efficiency plots (removes upper left labels)
- Core algorithm scaling analysis
- Advanced algorithm scaling analysis (execution time & memory)

### Scaling Analysis
- Core algorithms: Basic, Vectorized, Parallel
- Advanced algorithms: All 6 implementations
- Combined execution time and memory scaling plots
- Clean visualization without number annotations