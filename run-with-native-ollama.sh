#!/bin/bash
# Run the demo with native Ollama (Metal GPU support) instead of Docker Ollama

set -e

echo "=========================================="
echo "Running with Native Ollama (Metal GPU)"
echo "=========================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo ""
    echo "Install Ollama natively:"
    echo "  brew install ollama"
    echo "  OR download from https://ollama.com/download"
    echo ""
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama service..."
    ollama serve &
    OLLAMA_PID=$!
    echo "Ollama started (PID: $OLLAMA_PID)"
    echo "Waiting for Ollama to be ready..."
    sleep 3
else
    echo "Ollama is already running"
fi

# Pull base model if not present
echo ""
echo "Checking for base model..."
if ! ollama list | grep -q "llama3.1"; then
    echo "Pulling llama3.1 model (this may take a while)..."
    ollama pull llama3.1
else
    echo "Base model (llama3.1) already available"
fi

# Stop existing containers
echo ""
echo "Stopping existing containers..."
docker compose stop ollama web 2>/dev/null || true
docker compose rm -f ollama web 2>/dev/null || true

# Build web container
echo ""
echo "Building web container..."
docker compose build web

# Start only the web container (Ollama runs natively)
# Override OLLAMA_HOST to point to native Ollama
echo ""
echo "Starting web container (connecting to native Ollama at host.docker.internal:11434)..."
docker compose run -d \
    --name training-web \
    --rm=false \
    -p 8000:5000 \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/training_data:/app/training_data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/alice_in_wonderland.txt:/app/alice_in_wonderland.txt:ro" \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    web

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Web UI: http://localhost:8000"
echo ""
echo "Ollama is running natively (Metal GPU enabled)"
echo "Web service is running in Docker"
echo ""
echo "To stop:"
echo "  docker compose stop web"
echo "  ollama stop  # or kill the Ollama process"
echo ""

