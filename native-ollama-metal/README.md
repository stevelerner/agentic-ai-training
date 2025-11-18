# Native Ollama Metal Version

This is the native macOS version of the Agentic AI Training Demo, optimized for Metal GPU performance.

## Why This Version?

Docker Desktop on macOS doesn't support GPU passthrough, resulting in slow CPU-only inference. This native version uses Metal GPU acceleration for 2-5x better performance.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4) or Intel with AMD GPU
- Ollama installed: `brew install ollama`
- Python 3.8+

## Quick Start

```bash
./run.sh
```

This will:
1. Check and start Ollama (or show install instructions)
2. Pull the llama3.1 base model if needed
3. Create Python virtual environment
4. Install dependencies
5. Start the Flask server at http://localhost:5001

Stop with Ctrl+C

## Manual Setup

```bash
# Start Ollama
ollama serve

# Pull base model
ollama pull llama3.1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start server
python3 server.py
```

## Usage

Open http://localhost:5001 and:

1. Click "Prepare Training Data" to extract Mad Hatter dialogue
2. Click "Create Trained Model" to build the trained model
3. Use "Model Comparison" to compare base vs trained responses
4. Try the suggested prompts or create your own

## Scripts

- `./run.sh` - One-command setup and start
- `./verify.sh` - Check all dependencies and requirements
- `./cleanup.sh` - Remove generated files, keep Ollama and models
- `./create-model.sh` - Manually create trained model (usually done via web UI)

## Project Structure

```
native-ollama-metal/
├── server.py              # Flask server with agent logic
├── training.py            # Model training via Modelfiles
├── data_processor.py      # Dialogue extraction
├── requirements.txt       # Python dependencies
├── run.sh                 # Quick start script
├── create-model.sh        # Model creation helper
├── verify.sh              # Dependency checker
├── cleanup.sh             # Cleanup script
├── alice_in_wonderland.txt # Training data source
├── templates/             # Web UI
├── training_data/         # Generated training examples
├── checkpoints/           # Generated modelfiles
└── outputs/               # Agent outputs
```

## Differences from Docker Version

- Ollama runs natively (not in Docker)
- Uses Metal GPU for acceleration
- Server runs on port 5001 (vs 8000 in Docker, avoiding macOS AirPlay on 5000)
- No Docker dependencies
- Faster inference (20-50+ tokens/sec vs 5-10)

## Troubleshooting

**Ollama not found:**
```bash
brew install ollama
```

**Model not found:**
```bash
ollama list
ollama pull llama3.1
```

**Port in use:**
```bash
# Change port with environment variable
PORT=5002 python3 server.py
```

**Check setup:**
```bash
./verify.sh
```

**Cleanup:**
```bash
./cleanup.sh
```
Removes:
- Python virtual environment (venv/)
- Generated training data
- Generated checkpoints
- Generated outputs
- Python cache files

Preserves:
- Ollama and all models
- Source code and templates

Note: If you're in a venv, it will skip venv removal. Deactivate first if you want to remove it.

## Performance

Typical generation speeds on Apple Silicon:
- M1/M2: 20-30 tokens/second
- M3/M4: 30-50+ tokens/second

Compared to Docker (CPU only): 5-10 tokens/second

## What It Demonstrates

See the main README in the parent directory for a complete explanation of:
- Agentic AI with ReAct pattern
- Model training concepts
- Tool calling
- Evaluation metrics

This native version provides the same functionality with significantly better performance.
