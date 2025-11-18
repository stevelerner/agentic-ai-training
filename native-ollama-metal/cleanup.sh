#!/bin/bash
# Cleanup script for native-ollama-metal demo
# Removes generated files and virtual environment but leaves Ollama in place

# Don't use set -e to prevent any exit issues
set +e

echo "=========================================="
echo "Cleanup Native Ollama Metal Demo"
echo "=========================================="
echo ""

# Check if in virtual environment and skip venv deletion if so
SKIP_VENV=false
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Note: Virtual environment is currently active."
    echo "      Skipping venv removal (would break your terminal)."
    echo "      Run 'deactivate' first, then run cleanup.sh again."
    echo ""
    SKIP_VENV=true
fi

# Remove Python virtual environment
if [ -d "venv" ] && [ "$SKIP_VENV" = false ]; then
    echo "Removing Python virtual environment..."
    rm -rf venv
    echo "  [OK] Removed venv/"
elif [ "$SKIP_VENV" = true ]; then
    echo "Skipping venv removal (currently active)"
fi

# Remove Python cache
if [ -d "__pycache__" ]; then
    echo "Removing Python cache..."
    rm -rf __pycache__
    echo "  [OK] Removed __pycache__/"
fi

# Clean training data (keep directory)
if [ -d "training_data" ]; then
    echo "Cleaning training data..."
    rm -f training_data/*.jsonl
    echo "  [OK] Cleaned training_data/"
fi

# Clean checkpoints (keep directory)
if [ -d "checkpoints" ]; then
    echo "Cleaning checkpoints..."
    rm -f checkpoints/*.modelfile
    echo "  [OK] Cleaned checkpoints/"
fi

# Clean outputs (keep directory)
if [ -d "outputs" ]; then
    echo "Cleaning outputs..."
    rm -f outputs/*
    echo "  [OK] Cleaned outputs/"
fi

# Clean any .pyc files
echo "Cleaning .pyc files..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "  [OK] Cleaned .pyc files"

echo ""
echo "=========================================="
echo "Cleanup Complete"
echo "=========================================="
echo ""
echo "Cleaned:"
echo "  - Python virtual environment"
echo "  - Generated training data"
echo "  - Generated checkpoints"
echo "  - Generated outputs"
echo "  - Python cache files"
echo ""
echo "Preserved:"
echo "  - Ollama (still running)"
echo "  - Ollama models (llama3.1, mad-hatter if created)"
echo "  - Source code and templates"
echo ""
if [ "$SKIP_VENV" = true ]; then
    echo "To finish cleanup:"
    echo "  1. Run: deactivate"
    echo "  2. Run: ./cleanup.sh again"
    echo ""
fi
echo "To remove Ollama models:"
echo "  ollama rm mad-hatter"
echo "  ollama rm llama3.1"
echo ""

