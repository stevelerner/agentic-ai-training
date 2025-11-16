#!/bin/bash
# Cleanup script for training demo

set -e

echo "=========================================="
echo "Cleaning up AI Training Demo"
echo "=========================================="
echo ""

# Stop Docker containers
echo "Stopping Docker containers..."
docker compose down --rmi local 2>/dev/null || true

# Remove volumes
echo "Removing Docker volumes..."
docker volume rm training-demo_ollama_data 2>/dev/null || true

# Check if native Ollama is running and ask about stopping it
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo ""
        echo "Native Ollama is running."
        read -p "Stop native Ollama? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping native Ollama..."
            ollama stop 2>/dev/null || pkill -f ollama || true
            echo "Native Ollama stopped."
        else
            echo "Native Ollama left running."
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Note: Training data, checkpoints, and outputs are preserved."
echo "To remove them, delete:"
echo "  - training_data/"
echo "  - checkpoints/"
echo "  - outputs/"
echo ""
echo "Note: Ollama image (ollama/ollama:latest) is preserved to avoid re-downloading."
echo ""
echo "To rebuild and start:"
echo "  ./run.sh  (Docker Ollama)"
echo "  ./run-with-native-ollama.sh  (Native Ollama with Metal GPU)"
echo ""

