#!/bin/bash

# Setup script for Federated Learning Financial Forecasting Demo

echo "=========================================="
echo "Federated Learning Demo - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Docker is installed
echo ""
echo "Checking Docker installation..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version)
    echo "✓ Docker found: $docker_version"
    DOCKER_AVAILABLE=true
else
    echo "✗ Docker not found"
    echo ""
    echo "To install Docker Desktop on macOS:"
    echo "1. Visit: https://www.docker.com/products/docker-desktop/"
    echo "2. Download and install Docker Desktop"
    echo "3. Launch Docker Desktop from Applications"
    echo "4. Run this script again"
    echo ""
    DOCKER_AVAILABLE=false
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download financial data:"
echo "   python data/download_data.py"
echo ""
echo "2. Run the Streamlit app:"
echo "   streamlit run app.py"
echo ""
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "3. Or use Docker:"
    echo "   docker-compose up --build"
fi
echo ""

