#!/bin/bash

# Matrix Multiplication Benchmark Runner
# Usage: ./run_benchmarks.sh [test_type]
# Test types: verify, quick, full, sparse, all

PROJECT_DIR="/home/rgr/Documents/ULPGC/BD/BigData_ULPGC_IndividualAssignments/Task_2"
BINARY="matrix_benchmark"

cd "$PROJECT_DIR"

# Check if binary exists, build if not
if [ ! -f "$BINARY" ]; then
    echo "Building project..."
    make clean && make
    if [ $? -ne 0 ]; then
        echo "Build failed!"
        exit 1
    fi
fi

# Default to verification if no argument provided
TEST_TYPE=${1:-verify}

echo "Running matrix multiplication benchmarks..."
echo "Test type: $TEST_TYPE"
echo "================================================"

case $TEST_TYPE in
    "verify")
        ./$BINARY --verify
        ;;
    "quick")
        ./$BINARY --quick
        ;;
    "full")
        ./$BINARY --full
        ;;
    "sparse")
        ./$BINARY --sparse
        ;;
    "all")
        ./$BINARY --all
        ;;
    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Available options: verify, quick, full, sparse, all"
        ./$BINARY --help
        exit 1
        ;;
esac

echo "================================================"
echo "Benchmark complete!"