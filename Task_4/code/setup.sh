#!/bin/bash

echo "Setting up Distributed Matrix Multiplication Environment"
echo "======================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if Java is available
if ! command -v java &> /dev/null; then
    echo "❌ Java not found. Please install Java 11+"
    exit 1
fi

# Check if Maven is available
if ! command -v mvn &> /dev/null; then
    echo "❌ Maven not found. Please install Maven 3.6+"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup Python environment
echo ""
echo "Setting up Python environment..."
cd python
pip install -r requirements.txt
pip install -r benchmark_requirements.txt
echo "✅ Python dependencies installed"

# Setup Java environment
echo ""
echo "Setting up Java environment..."
cd ../java
mvn clean compile
echo "✅ Java project compiled"

cd ..

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Start Hazelcast server: hz start"
echo "2. Run benchmarks: python benchmarks/benchmark_comparison.py"
echo "3. Or run individual implementations:"
echo "   - Python: cd python && python production/matrix_multiplier.py"
echo "   - Java: cd java && mvn exec:java"