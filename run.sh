#!/bin/bash
# Quick start script for training demo

echo "Starting AI Training Demo..."

# Check for existing containers
if docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qE "^training-ollama$|^training-web$"; then
    echo ""
    echo "Error: Containers already exist. Please run cleanup first:"
    echo "  ./cleanup.sh"
    echo ""
    exit 1
fi

# Start containers
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
echo "Setup complete!"
echo "Open your browser to: http://localhost:8000"
echo ""
echo "To stop: docker compose down"

