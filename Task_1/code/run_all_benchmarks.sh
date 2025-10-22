#!/bin/bash

# Master benchmark runner script
# Runs matrix multiplication benchmarks for Python, Java, and C

echo "Matrix Multiplication Performance Comparison"
echo "============================================"
echo "Languages: Python, Java, C"
echo "Matrix sizes: 64x64, 128x128, 256x256, 512x512, 1024x1024"
echo "Runs per test: 5"
echo ""

# Create results directory
mkdir -p results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$(pwd)/results/benchmark_results_${TIMESTAMP}.csv"

# Initialize CSV file with header
echo "Language,Matrix_Size,Run_Number,Execution_Time_Seconds,Memory_Usage_MB" > "$RESULTS_FILE"

# Check if required tools are available
echo "Checking dependencies..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Python not found! Please install Python 3."
    exit 1
fi

# Check Java
if ! command -v javac &> /dev/null; then
    echo "Java compiler (javac) not found! Please install JDK."
    exit 1
fi

if ! command -v java &> /dev/null; then
    echo "Java runtime (java) not found! Please install JRE."
    exit 1
fi

# Check GCC
if ! command -v gcc &> /dev/null; then
    echo "GCC compiler not found! Please install GCC."
    exit 1
fi

echo "All dependencies found."
echo ""

# Run Python benchmark
echo "=== PYTHON BENCHMARK ==="
cd python
$PYTHON_CMD benchmark.py "$RESULTS_FILE"
cd ..

echo ""
echo "=== JAVA BENCHMARK ==="

# Compile Java files
cd java
echo "Compiling Java files..."
javac *.java
if [ $? -eq 0 ]; then
    echo "Java compilation successful."
    java MatrixBenchmark "$RESULTS_FILE"
else
    echo "Java compilation failed!"
    exit 1
fi
cd ..

echo ""
echo "=== C BENCHMARK ==="

# Compile and run C benchmark
cd c
echo "Compiling C files..."
make clean > /dev/null 2>&1
make
if [ $? -eq 0 ]; then
    echo "C compilation successful."
    ./matrix_benchmark "$RESULTS_FILE"
else
    echo "C compilation failed!"
    exit 1
fi
cd ..

echo ""
echo "=== GENERATING SUMMARY ==="
echo "Creating summary statistics from detailed results..."
$PYTHON_CMD generate_summary.py "$RESULTS_FILE"

echo ""
echo "=== SUMMARY ==="
echo ""
echo "Benchmark completed successfully!"
echo "Results saved to: $RESULTS_FILE"
echo "Summary saved to: ${RESULTS_FILE%.csv}_summary.csv"
echo ""
echo "To run individual benchmarks:"
echo "  Python: cd python && $PYTHON_CMD benchmark.py"
echo "  Java:   cd java && javac *.java && java MatrixBenchmark"
echo "  C:      cd c && make run"
echo ""
echo "To generate visualizations: python3 create_graphs.py"