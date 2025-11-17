#!/bin/bash
# Helper script to create the trained model in native Ollama
# Run this from the host machine after preparing training data

set -e

MODEL_NAME="mad-hatter"
MODELFILE_PATH="checkpoints/${MODEL_NAME}.modelfile"

if [ ! -f "$MODELFILE_PATH" ]; then
    echo "Error: Modelfile not found at $MODELFILE_PATH"
    echo "Please prepare training data first in the web UI"
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed."
    echo "Install with: brew install ollama"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Error: Ollama is not running."
    echo "Start it with: ollama serve"
    exit 1
fi

echo "Creating model in native Ollama..."
ollama create "${MODEL_NAME}" -f "${MODELFILE_PATH}"

echo ""
echo "Model '${MODEL_NAME}' created successfully in native Ollama!"
echo "You can now use it in the web UI."

