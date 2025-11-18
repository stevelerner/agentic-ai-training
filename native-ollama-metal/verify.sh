#!/bin/bash
# Verification script to check if all dependencies are met

echo "=========================================="
echo "Verifying Setup"
echo "=========================================="
echo ""

# Check Ollama
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "[OK] Ollama is installed"
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[OK] Ollama is running"
        
        # Check for base model
        if ollama list | grep -q "llama3.1"; then
            echo "[OK] Base model (llama3.1) is available"
        else
            echo "[WARN] Base model (llama3.1) not found"
            echo "       Run: ollama pull llama3.1"
        fi
    else
        echo "[WARN] Ollama is not running"
        echo "       Start it with: ollama serve"
    fi
else
    echo "[FAIL] Ollama is not installed"
    echo "       Install with: brew install ollama"
fi

echo ""

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "[OK] Python is installed (version $PYTHON_VERSION)"
else
    echo "[FAIL] Python is not installed"
    echo "       Install with: brew install python3"
fi

echo ""

# Check required files
echo "Checking required files..."
REQUIRED_FILES=(
    "server.py"
    "training.py"
    "data_processor.py"
    "requirements.txt"
    "alice_in_wonderland.txt"
    "templates/index.html"
)

ALL_FILES_OK=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "[OK] $file"
    else
        echo "[FAIL] $file is missing"
        ALL_FILES_OK=false
    fi
done

echo ""

# Check directories
echo "Checking directories..."
REQUIRED_DIRS=(
    "training_data"
    "checkpoints"
    "outputs"
    "templates"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "[OK] $dir/"
    else
        echo "[FAIL] $dir/ is missing"
        ALL_FILES_OK=false
    fi
done

echo ""
echo "=========================================="

if [ "$ALL_FILES_OK" = true ]; then
    echo "All required files and directories present"
    echo ""
    echo "Ready to run! Execute: ./run.sh"
else
    echo "Some files or directories are missing"
    echo ""
    echo "Please check the setup and try again"
fi

echo "=========================================="
