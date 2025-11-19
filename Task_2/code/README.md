# Matrix Multiplication Optimizations

This project implements and benchmarks different matrix multiplication optimization techniques.

## Algorithms

1. **Basic** - Standard O(n³) algorithm
2. **Loop Unrolling** - 4-way unrolled inner loops  
3. **Cache Blocking** - Block-based multiplication (64×64 blocks)
4. **Strassen** - Recursive divide-and-conquer O(n^2.807)
5. **Sparse CSR** - Compressed Sparse Row format

## Project Structure

```
Task_2/
├── src/
│   ├── cache_block.c        # Cache-blocked matrix multiplication
│   ├── loop_unroll.c        # Loop unrolling optimization
│   ├── matrix.c             # Basic matrix operations
│   ├── sparse_matrix.c      # Sparse matrix operations (CSR)
│   └── strassen.c           # Strassen algorithm implementation
├── include/
│   ├── benchmark.h          # Benchmark function declarations
│   ├── matrix.h             # Matrix structure and function declarations
│   └── sparse_matrix.h      # Sparse matrix structure and functions
├── benchmarks/
│   └── main.c               # Main benchmark entry point
├── Makefile       # Build system
└── README.md      # This file
```

## Building and Running

```bash
# Build optimized version
make

# Build debug version
make debug

# Run benchmarks
make run

# Clean build artifacts
make clean
```

## Benchmarks

Tests include:
- Execution times (256×256 to 2048×2048)
- Memory usage and efficiency 
- Maximum matrix sizes (1s timeout)
- Sparse vs dense performance (50%, 70%, 90%, 95% sparsity)

## Visualization

Generate performance plots:

```bash
python3 plot.py
```

Creates plots for:
- Dense execution times and speedups
- Memory usage and efficiency
- Sparse matrix comparisons
- Performance scaling analysis

## Requirements

- GCC compiler
- Linux system (tested on Ubuntu/similar)
- Sufficient RAM for large matrix testing
- Python 3 with matplotlib, pandas, seaborn (for visualization)