#!/bin/bash
# Setup script for Python plotting dependencies

echo "🐍 Setting up Python plotting environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3 first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed successfully!"
    echo ""
    echo "🎨 Testing plotting system..."
    
    # Check if CSV data exists
    if [ -d "results" ] && [ "$(ls -A results/*.csv 2>/dev/null)" ]; then
        echo "📊 Found benchmark data, generating comprehensive plots..."
        python3 plotting.py
        echo "✅ Comprehensive visualization suite generated!"
        echo "   📁 Check plots/ directory for results"
    else
        echo "ℹ️  No benchmark data found. Run benchmarks first:"
        echo "   ./build.sh && cd build && java benchmark.PerformanceBenchmark"
        echo "   Then run: python3 plotting.py"
    fi
    
    echo ""
    echo "🎯 Setup complete! Available commands:"
    echo "   📊 Generate plots: python3 plotting.py"
    echo "   🏃 Run benchmarks: ./build.sh && cd build && java benchmark.PerformanceBenchmark"
else
    echo "❌ Failed to install Python dependencies."
    echo "   Manual install: source .venv/bin/activate && pip install matplotlib seaborn pandas numpy"
fi