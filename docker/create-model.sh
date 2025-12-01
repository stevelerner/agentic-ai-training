#!/bin/bash
# Helper script to create the trained model from modelfile
# Run this from the host machine after preparing training data

set -e

MODEL_NAME="mad-hatter"
CONTAINER_NAME="training-ollama"
MODELFILE_PATH="checkpoints/${MODEL_NAME}.modelfile"

if [ ! -f "$MODELFILE_PATH" ]; then
    echo "Error: Modelfile not found at $MODELFILE_PATH"
    echo "Please prepare training data first in the web UI"
    exit 1
fi

echo "Copying modelfile to container..."
docker cp "$MODELFILE_PATH" "${CONTAINER_NAME}:/root/.ollama/${MODEL_NAME}.modelfile"

echo "Creating model in Ollama..."
docker exec "${CONTAINER_NAME}" ollama create "${MODEL_NAME}" -f "/root/.ollama/${MODEL_NAME}.modelfile"

echo "Model '${MODEL_NAME}' created successfully!"
echo "You can now use it in the web UI."

