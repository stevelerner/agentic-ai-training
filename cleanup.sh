#!/bin/bash
# Cleanup script for training demo

echo "Stopping and removing containers..."

# Stop and remove containers by name (in case docker compose doesn't catch them)
docker stop training-ollama training-web 2>/dev/null || true
docker rm training-ollama training-web 2>/dev/null || true

# Also use docker compose to clean up
docker compose down --rmi local 2>/dev/null || true

echo ""
echo "Cleanup complete!"
echo "Note: Ollama image (ollama/ollama:latest) is preserved to avoid re-downloading."
echo ""
echo "To remove volumes (deletes models and training data):"
echo "  docker compose down -v"
echo ""
echo "To rebuild and start:"
echo "  ./run.sh"
echo ""

