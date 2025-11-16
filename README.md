# AI Training Demo

A minimal demonstration of agentic AI and model training using Alice in Wonderland.

## What This Demonstrates

**Agentic AI:**
- ReAct pattern (Reason → Act → Observe loop)
- Tool calling (calculate, save_file)
- Model switching (base vs trained)

**Model Training:**
- Data extraction from text
- Character-specific fine-tuning
- Before/after model comparison

## Architecture

- **Ollama container**: LLM inference (base and trained models)
  - Uses Metal GPU acceleration on Apple Silicon (automatic)
  - Faster inference with GPU support
- **Web container**: Agent logic, training pipeline, web UI
- **Volumes**: Training data, checkpoints, outputs

## Quick Start

**Prerequisites:**
- Docker Desktop running
- 8GB+ RAM available
- macOS with Apple Silicon (for Metal GPU acceleration)

**Steps:**

1. Start containers:
```bash
docker compose up -d --build
```

2. Pull base model:
```bash
docker exec training-ollama ollama pull llama3.1
```

3. Open browser:
```
http://localhost:8000
```

4. In the web UI:
   - Click "Prepare Training Data" to extract Mad Hatter dialogue
   - Click "Create Trained Model" to train the model
   - Use "Agent Query" to test tool calling
   - Use "Model Comparison" to see base vs trained differences

## How It Works

### Training Process

1. **Data Extraction**: Extracts Mad Hatter dialogue from Alice in Wonderland
   - Finds dialogue attributed to "Hatter"
   - Focuses on tea party scene (Chapter VII)
   - Formats as instruction-response pairs

2. **Model Creation**: Creates a new Ollama model with:
   - System prompt defining Mad Hatter character
   - Base model (llama3.1) as foundation
   - Character-specific behavior

3. **Training Format**: Uses Ollama Modelfile approach
   - System prompt for character definition
   - Template-based fine-tuning
   - Simplified for demo purposes

### Agentic AI

The agent implements ReAct pattern:
- Receives user query
- Decides if tools are needed
- Executes tools (calculate, save_file)
- Processes results
- Returns final answer

Can switch between base and trained models to show differences.

## Files

- `server.py`: Web server, agent logic, training API
- `training.py`: Model creation and training functions
- `data_processor.py`: Extract character dialogue from text
- `docker-compose.yml`: Container orchestration
- `Dockerfile`: Web service container
- `templates/index.html`: Web UI
- `alice_in_wonderland.txt`: Source text for training

## Example Queries

**Agent Query:**
- "Calculate 25 * 4 and save the result to result.txt"
- "What is 100 divided by 5?"

**Model Comparison:**
- "What time is it?"
- "Why is a raven like a writing-desk?"
- "Tell me about yourself"

## Training Data

Training data is extracted from:
- Chapter VII: A Mad Tea-Party (primary source)
- Other chapters with Hatter dialogue
- Saved to `training_data/mad_hatter_training.jsonl`

## Model Comparison

The trained model (mad-hatter) should:
- Reference time/tea time obsessively
- Speak in absurd, nonsensical ways
- Ask riddles
- Show philosophical but illogical reasoning

Base model (llama3.1) responds normally.

## Metal GPU Support

**Apple Silicon (M1/M2/M3):**
- Metal GPU acceleration is automatically enabled
- Ollama detects and uses Metal for faster inference
- No additional configuration required
- Significantly faster than CPU-only execution
- Configured for `linux/arm64` platform in docker-compose.yml

**Intel Mac:**
- Remove `platform: linux/arm64` from docker-compose.yml if needed
- Will run on CPU (slower but functional)

**Verifying Metal Usage:**
```bash
docker logs training-ollama | grep -i metal
```

Ollama will automatically use Metal when available on macOS.

## Limitations

This is a simplified training demo:
- Uses system prompt approach (not full fine-tuning)
- Limited training data (single character, single book)
- Modelfile approach is basic

For production training, use:
- LoRA/QLoRA with transformers
- Larger datasets
- Proper fine-tuning pipelines

## Troubleshooting

**Model not found:**
```bash
docker exec training-ollama ollama list
docker exec training-ollama ollama pull llama3.1
```

**Training fails:**
- Check base model is available
- Verify training data was prepared
- Check container logs: `docker logs training-web`

**Agent not responding:**
- Verify Ollama is running: `docker ps | grep ollama`
- Check logs: `docker logs training-ollama`

**Metal GPU not working:**
- Ensure Docker Desktop is using Apple Silicon (arm64)
- Check Docker Desktop settings for GPU support
- Verify platform: `docker compose config | grep platform`
- Ollama will fall back to CPU if Metal unavailable

## Cleanup

Stop containers:
```bash
docker compose down
```

Remove volumes (deletes models):
```bash
docker compose down -v
```

