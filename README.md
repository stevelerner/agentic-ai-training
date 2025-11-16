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

**Docker Setup (Default):**
- **Ollama container**: LLM inference (base and trained models)
  - Runs on CPU only (Docker Desktop on macOS doesn't support GPU passthrough)
  - Slower but fully containerized
- **Web container**: Agent logic, training pipeline, web UI
- **Volumes**: Training data, checkpoints, outputs

**Native Ollama Setup (Metal GPU):**
- **Native Ollama**: Runs directly on macOS
  - Uses Metal GPU on Apple Silicon for faster inference
  - Accessible via `host.docker.internal:11434`
- **Web container**: Agent logic, training pipeline, web UI
  - Connects to native Ollama instead of Docker Ollama
- **Volumes**: Training data, checkpoints, outputs

## Quick Start

**Prerequisites:**
- Docker Desktop running
- 8GB+ RAM available
- macOS with Apple Silicon (for Metal GPU acceleration - optional)

**Option 1: Quick Start with Docker Ollama (CPU only)**

```bash
./run.sh
```

This will:
- Start Ollama in Docker (CPU only - slower)
- Start the web UI
- Pull the base model if needed
- Open http://localhost:8000

**Option 2: Native Ollama with Metal GPU (Recommended for Apple Silicon)**

For faster inference with Metal GPU acceleration:

```bash
./run-with-native-ollama.sh
```

This will:
- Start Ollama natively on macOS (Metal GPU enabled)
- Start only the web container in Docker
- Connect web container to native Ollama
- Provide significantly faster inference

**Manual Setup:**

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
   - Click "Check Models" to verify models are available
   - Click "Check GPU/Metal" to see GPU status
   - Click "Prepare Training Data" to extract Mad Hatter dialogue
   - Click "Create Trained Model" to create the modelfile
   - **For Docker Ollama**: Run `./create-model.sh` from host
   - **For Native Ollama**: Run `./create-model-native.sh` from host
   - Use "Model Comparison" to see base vs trained differences
   - Use "Agent Query" to test tool calling with suggested questions

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

**Core Files:**
- `server.py`: Web server, agent logic, training API, GPU detection
- `training.py`: Model creation and training functions
- `data_processor.py`: Extract character dialogue from text
- `docker-compose.yml`: Container orchestration
- `Dockerfile`: Web service container
- `templates/index.html`: Web UI with suggested questions and tool usage display

**Scripts:**
- `run.sh`: Quick start with Docker Ollama (CPU only)
- `run-with-native-ollama.sh`: Start with native Ollama (Metal GPU)
- `create-model.sh`: Create trained model in Docker Ollama (run from host)
- `create-model-native.sh`: Create trained model in native Ollama (run from host)
- `cleanup.sh`: Clean up containers and volumes

**Data:**
- `alice_in_wonderland.txt`: Source text for training
- `training_data/`: Extracted training examples (JSONL format)
- `checkpoints/`: Generated modelfiles
- `outputs/`: Agent-generated files

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

**Important:** Docker Desktop on macOS does **not support GPU passthrough**. Ollama running in Docker containers will always use CPU, even on Apple Silicon.

**To Enable Metal GPU:**

Use the native Ollama setup for Metal GPU acceleration on Apple Silicon:

1. **Install Ollama natively** (if not already installed):
   ```bash
   brew install ollama
   # OR download from https://ollama.com/download
   ```

2. **Run with native Ollama:**
   ```bash
   ./run-with-native-ollama.sh
   ```

This script:
- Starts Ollama natively on macOS (Metal GPU enabled automatically)
- Runs only the web container in Docker
- Connects the web container to native Ollama via `host.docker.internal:11434`
- Provides significantly faster inference (2-5x speedup on Apple Silicon)

**Verifying GPU Usage:**

In the web UI:
- Click "Check GPU/Metal" button to see current GPU status
- Shows "Metal GPU (native Ollama)" when using native Ollama
- Shows "CPU (Docker container)" when using Docker Ollama

From command line:
```bash
# Check if Metal is being used (native Ollama)
ollama ps

# Monitor GPU usage
# Open Activity Monitor > Window > GPU History
# You should see GPU activity when running queries
```

**Performance:**
- **Docker Ollama (CPU)**: ~5-15 seconds per query
- **Native Ollama (Metal GPU)**: ~2-5 seconds per query (Apple Silicon)

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

**Model not found in native Ollama:**
- If you created the model in Docker Ollama, you need to create it in native Ollama too
- Run `./create-model-native.sh` from the host machine
- Verify: `ollama list` should show `mad-hatter`

**Agent not responding:**
- **Docker Ollama**: Verify container is running: `docker ps | grep ollama`
- **Docker Ollama**: Check logs: `docker logs training-ollama`
- **Native Ollama**: Verify Ollama is running: `ollama ps` or check if port 11434 is accessible
- **Native Ollama**: Check if web container can reach it: `docker exec training-web curl -s http://host.docker.internal:11434/api/tags`

**Metal GPU not working:**
- Docker Ollama always runs on CPU (Docker Desktop limitation)
- Use `./run-with-native-ollama.sh` for Metal GPU support
- Verify native Ollama is running: `ollama ps`
- Check GPU status in web UI: Click "Check GPU/Metal"
- Monitor GPU usage: Activity Monitor > Window > GPU History

## Cleanup

**Stop Docker setup:**
```bash
docker compose down
```

**Stop native Ollama setup:**
```bash
docker compose stop web
docker compose rm -f web
ollama stop  # or kill the Ollama process
```

**Remove volumes (deletes models):**
```bash
docker compose down -v
```

**Full cleanup (including images):**
```bash
./cleanup.sh
```

