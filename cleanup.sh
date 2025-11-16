#!/bin/bash
# Cleanup script for training demo

echo "Stopping containers and removing web image (preserves Ollama image)..."
# --rmi local only removes locally built images (web service), not pulled images (ollama/ollama:latest)
# This prevents re-downloading the Ollama image while allowing web container rebuilds
docker compose down --rmi local

echo ""
echo "Cleanup complete!"
echo "Note: Ollama image (ollama/ollama:latest) is preserved to avoid re-downloading."
echo ""
echo "To remove volumes (deletes models and training data):"
echo "  docker compose down -v"
echo ""
echo "To rebuild and start:"
echo "  ./run.sh"

