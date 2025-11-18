#!/bin/bash
# Helper script to create the trained model in native Ollama

set -e

MODEL_NAME="mad-hatter"
MODELFILE_PATH="checkpoints/${MODEL_NAME}.modelfile"

echo "=========================================="
echo "Creating Trained Model: ${MODEL_NAME}"
echo "=========================================="
echo ""

if [ ! -f "$MODELFILE_PATH" ]; then
    echo "Error: Modelfile not found at $MODELFILE_PATH"
    echo ""
    echo "Please prepare training data first:"
    echo "  1. Open http://localhost:5001"
    echo "  2. Click 'Prepare Training Data'"
    echo "  3. Click 'Create Trained Model'"
    echo ""
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo ""
    echo "Install with:"
    echo "  brew install ollama"
    echo "  OR download from https://ollama.com/download"
    echo ""
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Error: Ollama is not running."
    echo ""
    echo "Start it with:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo "Creating model in native Ollama..."
echo ""
ollama create "${MODEL_NAME}" -f "${MODELFILE_PATH}"

echo ""
echo "=========================================="
echo "Success!"
echo "=========================================="
echo ""
echo "Model '${MODEL_NAME}' created successfully!"
echo "You can now use it in the web UI for comparisons."
echo ""
