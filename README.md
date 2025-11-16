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
  - Metal GPU acceleration on Apple Silicon (if enabled in Docker Desktop)
  - Falls back to CPU if GPU not available
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
   - Click "Create Trained Model" to create the modelfile
   - Run the helper script from host: `./create-model.sh`
   - Use "Agent Query" to test tool calling
   - Use "Model Comparison" to see base vs trained differences

## Agentic AI vs LLM Chat

**LLM Chat:**
- Single request → single response
- No actions beyond text generation
- Cannot interact with external systems
- Example: "What time is it?" → "It's currently 3:45 PM"

**Agentic AI (This Demo):**
- Multi-step reasoning loop (ReAct pattern)
- Decides when to use tools autonomously
- Executes actions (calculate, save files)
- Observes results and adapts
- Example: "Ask the Mad Hatter what time it is, calculate 6 o'clock in minutes, and save the result" → Queries trained model → Gets "It's always six o'clock!" → Calculates 6 * 60 = 360 → Saves to file → Confirms completion

**Key Difference:**
- Agentic: Can take actions, use tools, iterate until task complete
- LLM Chat: Only generates text responses

## How Training Works

**Step 1: Data Extraction**
- Scans Alice in Wonderland text for Mad Hatter dialogue
- Extracts quotes attributed to "Hatter" character
- Focuses on tea party scene (Chapter VII) for most examples
- Formats as instruction-response pairs for training

**Step 2: Model Creation**
- Creates Ollama Modelfile with system prompt
- Defines Mad Hatter character traits (time-obsessed, absurd, riddles)
- Uses base model (llama3.1) as foundation
- Applies character-specific behavior via system prompt

**Step 3: Training Result**
- New model "mad-hatter" available in Ollama
- Responds in character style when queried
- Shows before/after difference vs base model
- Demonstrates how training changes model behavior

**Training Approach:**
- Uses Ollama Modelfile (system prompt method)
- Simplified for demo (not full parameter fine-tuning)
- Fast to create, easy to understand
- Shows concept of model customization

## Files

- `server.py`: Web server, agent logic, training API
- `training.py`: Model creation and training functions
- `data_processor.py`: Extract character dialogue from text
- `docker-compose.yml`: Container orchestration
- `Dockerfile`: Web service container
- `templates/index.html`: Web UI
- `alice_in_wonderland.txt`: Source text for training
- `create-model.sh`: Helper script to create trained model from modelfile

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
- Metal GPU acceleration may be available depending on Docker Desktop configuration
- Ollama will attempt to use Metal for faster inference
- Falls back to CPU if Metal unavailable in container
- Configured for `linux/arm64` platform in docker-compose.yml
- `OLLAMA_NUM_GPU=1` environment variable is set

**Intel Mac:**
- Remove `platform: linux/arm64` from docker-compose.yml if needed
- Will run on CPU (slower but functional)

**Verifying GPU Usage:**
```bash
# Check if Metal/GPU is detected
docker logs training-ollama | grep -i "device\|gpu\|metal"

# Check current model status
docker exec training-ollama ollama ps

# Look for "device=CPU" vs "device=GPU" in logs
docker logs training-ollama | grep "device="
```

**Note:** Metal GPU access in Docker containers on macOS is **not available**. Docker Desktop on macOS does not support GPU passthrough. Ollama will always run on CPU inside Docker containers.

**To Enable Metal GPU:**

Run Ollama natively on macOS (outside Docker) for Metal GPU acceleration:

1. **Install Ollama natively:**
   ```bash
   brew install ollama
   # OR download from https://ollama.com/download
   ```

2. **Start Ollama natively:**
   ```bash
   ollama serve
   ```

3. **Use the native Ollama script:**
   ```bash
   ./run-with-native-ollama.sh
   ```

This script:
- Starts Ollama natively (Metal GPU enabled)
- Runs only the web container in Docker
- Connects the web container to native Ollama via `host.docker.internal`

**Verifying Metal GPU:**
```bash
# Check if Metal is being used (native Ollama)
ollama ps

# Monitor GPU usage
# Open Activity Monitor > Window > GPU History
```

The demo works on CPU, but Metal GPU provides significantly faster inference.

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

