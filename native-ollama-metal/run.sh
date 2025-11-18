#!/bin/bash
# Run the Agentic AI Training Demo with Native Ollama (Metal GPU)

set -e

echo "=========================================="
echo "Agentic AI Training Demo (Metal GPU)"
echo "=========================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo ""
    echo "Install Ollama first:"
    echo "  brew install ollama"
    echo "  OR download from https://ollama.com/download"
    echo ""
    echo "Then run this script again."
    exit 0
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo "Ollama started (PID: $OLLAMA_PID)"
    echo "  Waiting for Ollama to be ready..."
    sleep 3
else
    echo "Ollama is already running"
fi

# Check for base model
echo ""
echo "Checking for base model (llama3.1)..."
if ! ollama list | grep -q "llama3.1"; then
    echo "  Pulling llama3.1 model (this may take a while)..."
    ollama pull llama3.1
else
    echo "Base model (llama3.1) is available"
fi

# Check if Python venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
fi

# Activate venv and install dependencies
echo ""
echo "Installing Python dependencies..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "Dependencies installed"

# Start the Flask server
echo ""
echo "=========================================="
echo "Starting Flask Server"
echo "=========================================="
echo ""
echo "Web UI: http://localhost:5001"
echo ""
echo "Features:"
echo "  - Agentic AI with ReAct pattern"
echo "  - Model training (Mad Hatter character)"
echo "  - Model comparison with evaluation metrics"
echo "  - Metal GPU acceleration (native Ollama)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 server.py
