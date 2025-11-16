#!/bin/bash
# Cleanup script for training demo

echo "Stopping containers..."
docker compose down

echo ""
echo "To remove volumes (deletes models and data):"
echo "  docker compose down -v"

