#!/bin/bash
# Quick start script for training demo

set -e

echo "=========================================="
echo "AI Training Demo - Quick Start"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "This script starts Ollama in Docker (CPU only)."
echo ""
echo "For Metal GPU acceleration on Apple Silicon, use:"
echo "  ./run-with-native-ollama.sh"
echo ""
read -p "Continue with Docker Ollama (CPU)? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. Run ./run-with-native-ollama.sh for Metal GPU support."
    exit 0
fi

# Start containers
echo ""
echo "Starting Docker containers..."
docker compose up -d --build

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 5

# Check if base model exists
echo "Checking for base model..."
if ! docker exec training-ollama ollama list | grep -q "llama3.1"; then
    echo "Pulling base model (this may take a few minutes)..."
    docker exec training-ollama ollama pull llama3.1
else
    echo "Base model already available"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Web UI: http://localhost:8000"
echo ""
echo "Note: Ollama is running in Docker (CPU only)."
echo "For Metal GPU support, stop this and run: ./run-with-native-ollama.sh"
echo ""
echo "Next steps:"
echo "1. Open http://localhost:8000 in your browser"
echo "2. Click 'Prepare Training Data' to extract Mad Hatter dialogue"
echo "3. Click 'Create Trained Model' to create the mad-hatter model"
echo ""
echo "To stop: docker compose down"

