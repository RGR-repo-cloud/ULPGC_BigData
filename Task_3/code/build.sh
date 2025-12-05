#!/bin/bash

# Build script for the matrix multiplication benchmark project

echo "🔨 Building Matrix Multiplication Benchmark..."

# Create build directory if it doesn't exist
mkdir -p build
mkdir -p results
mkdir -p plots

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build/main build/benchmark

# Compile all Java files
echo "📦 Compiling Java source files..."
javac -d build src/main/*.java src/benchmark/*.java

# Check compilation success
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Available commands:"
    echo "  📊 Run benchmark:     cd build && java benchmark.PerformanceBenchmark"
    echo "  🎨 Generate plots:    python3 plotting.py"
    echo "  🔧 Setup plotting:    ./setup_plotting.sh"
    echo ""
    echo "📁 Output directories:"
    echo "  📈 Results:   results/"
    echo "  🎯 Plots:    plots/"
else
    echo "❌ Compilation failed!"
    echo "💡 Check Java source files for syntax errors"
    exit 1
fi