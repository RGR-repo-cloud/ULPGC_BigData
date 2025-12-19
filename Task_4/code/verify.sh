#!/bin/bash

echo "=========================================="
echo "Distributed Matrix Multiplication"
echo "System Verification"
echo "=========================================="
echo ""

# Check Hazelcast
echo "1. Checking Hazelcast server..."
if ps aux | grep -v grep | grep -q hazelcast; then
    echo "   ✓ Hazelcast is running"
    HAZELCAST_PID=$(ps aux | grep -v grep | grep hazelcast | awk '{print $2}' | head -1)
    echo "     PID: $HAZELCAST_PID"
else
    echo "   ✗ Hazelcast is NOT running"
    echo "     Start with: cd /home/rgr/Downloads/hazelcast-5.6.0-slim && nohup ./bin/hz start > /dev/null 2>&1 &"
fi

echo ""
echo "2. Checking Python environment..."
if [ -f "/home/rgr/Documents/ULPGC/BD/BigData_ULPGC_IndividualAssignments/Task_4/code/.venv/bin/python" ]; then
    echo "   ✓ Virtual environment exists"
    PYTHON_PATH="/home/rgr/Documents/ULPGC/BD/BigData_ULPGC_IndividualAssignments/Task_4/code/.venv/bin/python"
    PYTHON_VERSION=$($PYTHON_PATH --version 2>&1)
    echo "     $PYTHON_VERSION"
    
    # Check required packages
    if $PYTHON_PATH -c "import hazelcast" 2>/dev/null; then
        echo "   ✓ hazelcast-python-client installed"
    else
        echo "   ✗ hazelcast-python-client NOT installed"
    fi
    
    if $PYTHON_PATH -c "import numpy" 2>/dev/null; then
        echo "   ✓ numpy installed"
    else
        echo "   ✗ numpy NOT installed"
    fi
else
    echo "   ✗ Virtual environment not found"
fi

echo ""
echo "3. Checking Java environment..."
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -1)
    echo "   ✓ Java found: $JAVA_VERSION"
else
    echo "   ✗ Java NOT found"
fi

if command -v mvn &> /dev/null; then
    MVN_VERSION=$(mvn --version 2>&1 | head -1)
    echo "   ✓ Maven found: $MVN_VERSION"
else
    echo "   ✗ Maven NOT found"
fi

echo ""
echo "4. Checking project files..."
cd /home/rgr/Documents/ULPGC/BD/BigData_ULPGC_IndividualAssignments/Task_4/code

FILES=(
    "python/production/matrix_multiplier.py"
    "java/src/main/java/com/ulpgc/bigdata/DistributedMatrixMultiplier.java"
    "python/benchmarks/performance_benchmark.py"
    "benchmarks/benchmark_comparison.py"
    "README.md"
    "DESIGN.md"
    "REPORT_TEMPLATE.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file MISSING"
    fi
done

echo ""
echo "=========================================="
echo "Quick Test Commands"
echo "=========================================="
echo ""
echo "Test Python implementation:"
echo "  cd python && $PYTHON_PATH production/matrix_multiplier.py"
echo ""
echo "Test Java implementation:"
echo "  cd java && mvn exec:java -q"
echo ""
echo "Run benchmarks:"
echo "  cd python && $PYTHON_PATH benchmarks/performance_benchmark.py"
echo ""
echo "=========================================="