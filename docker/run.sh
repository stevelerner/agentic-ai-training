#!/bin/bash
# Quick start script for training demo

# Don't exit on errors - handle them gracefully
set +e

echo "Starting AI Training Demo..."

# Check for existing containers
if docker ps -a --format "{{.Names}}" 2>/dev/null | grep -qE "^training-ollama$|^training-web$"; then
    echo ""
    echo "Containers already exist. Please run cleanup first:"
    echo "  ./cleanup.sh"
    echo ""
    # Pause before exiting to keep terminal open
    read -p "Press Enter to exit..."
    exit 0
fi

# Start containers
echo "Starting Docker containers..."
docker compose up -d --build
if [ $? -ne 0 ]; then
    echo "Error: Failed to start containers"
    read -p "Press Enter to exit..."
    exit 1
fi

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
sleep 5

# Check if base model exists
echo "Checking for base model..."
# Check if model exists (grep returns non-zero when no match, which is expected)
docker exec training-ollama ollama list 2>/dev/null | grep -q "llama3.1"
model_exists=$?

if [ $model_exists -ne 0 ]; then
    echo "Pulling base model (this may take a few minutes)..."
    docker exec training-ollama ollama pull llama3.1
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to pull base model. You may need to pull it manually."
    fi
else
    echo "Base model already available"
fi

echo ""
echo "Setup complete!"
echo "Open your browser to: http://localhost:8000"
echo ""
echo "To stop: docker compose down"
echo ""
read -p "Press Enter to exit..."
exit 0

