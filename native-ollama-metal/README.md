# Native Ollama Metal Version (True Training Demo)

This is the native macOS version of the Agentic AI Training Demo, optimized for Metal GPU performance.
**Unlike the Docker version, this version demonstrates TRUE model fine-tuning (LoRA) using Apple's MLX framework.**

## Why This Version?

1.  **True Fine-Tuning**: Instead of just creating a Modelfile with a system prompt (like the Docker version), this version actually trains a LoRA adapter on the Llama 3 model using the `mlx-lm` library.
2.  **Performance**: Docker Desktop on macOS doesn't support GPU passthrough. This native version uses Metal GPU acceleration for 2-5x better performance.
3.  **Local Inference**: Uses MLX for efficient local inference with the fine-tuned adapters.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.8+
- Ollama installed (for the base model comparison): `brew install ollama`

## Quick Start

```bash
./run.sh
```

This will:
1.  Check dependencies (installing MLX, transformers, etc.)
2.  Start the Flask server at http://localhost:5001
3.  Open the web UI

**Note**: The first time you run training, it will download the base Llama 3 model (approx 2-3GB) which may take a few minutes.

## Usage

Open http://localhost:5001 and:

1.  **Prepare Data**: Click "Prepare Training Data" to extract Mad Hatter dialogue from Alice in Wonderland and format it for Llama 3 instruction tuning.
2.  **Train Model**: Click "Create Trained Model". **This now runs actual LoRA fine-tuning** for 100 iterations. You will see the loss decrease in the server logs.
3.  **Compare**: Use "Model Comparison" to chat with:
    - **Base Model**: Standard Llama 3.1 (via Ollama)
    - **Trained Model**: Llama 3.2 1B + LoRA Adapter (via MLX)

## Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (includes MLX)
pip install -r requirements.txt

# Start server
python3 server.py
```

## Project Structure

```
native-ollama-metal/
├── server.py              # Flask server with MLX integration
├── train_mlx.py           # [NEW] MLX LoRA training script
├── inference_mlx.py       # [NEW] MLX inference script
├── data_processor.py      # Data extraction & formatting
├── requirements.txt       # Dependencies (mlx, flask, etc.)
├── run.sh                 # Quick start script
├── alice_in_wonderland.txt # Training data source
├── training_data/         # Generated JSONL datasets
├── adapters/              # [NEW] Saved LoRA adapters
└── templates/             # Web UI
```

## How It Works

### 1. Data Preparation
`data_processor.py` extracts dialogue and formats it into the Llama 3 chat format:
```json
{"text": "<|begin_of_text|><|start_header_id|>system...<|end_header_id|>...<|start_header_id|>user...<|end_header_id|>...<|start_header_id|>assistant...<|end_header_id|>..."}
```

### 2. LoRA Fine-Tuning (`train_mlx.py`)
We use `mlx-lm` to train a Low-Rank Adapter (LoRA) on the `Llama-3.2-1B-Instruct` model.
- **Rank**: 8
- **Layers**: 16
- **Iterations**: 100 (fast demo)
- **Output**: `adapters/adapters.safetensors`

### 3. Inference (`inference_mlx.py`)
The server calls this script to load the base model + adapter and generate responses.

## Differences from Docker Version

| Feature | Docker Version | Native Metal Version |
|---------|---------------|----------------------|
| **Training Method** | System Prompt (Modelfile) | **LoRA Fine-Tuning (MLX)** |
| **GPU Support** | No (CPU only on Mac) | **Yes (Metal)** |
| **Model** | Llama 3.1 (Ollama) | Llama 3.2 1B (MLX) |
| **Complexity** | Low | Moderate |
| **Realism** | Simulation | **Real Training** |

## Troubleshooting

**Training is slow?**
- Ensure you are on Apple Silicon.
- The first run downloads the model; subsequent runs are faster.

**Ollama errors?**
- Ensure `ollama serve` is running for the Base Model comparison.

**Port in use?**
- The server runs on port 5001 by default to avoid conflicts with AirPlay (5000) or the Docker demo (8000).
