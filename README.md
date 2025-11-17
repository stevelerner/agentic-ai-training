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
- **Web container**: Agent logic, training pipeline, web UI
- **Volumes**: Training data, checkpoints, outputs

## Quick Start

**Prerequisites:**
- Docker Desktop running
- 8GB+ RAM available

```bash
./run.sh
```

This will:
- Start Ollama in Docker
- Start the web UI
- Pull the base model if needed
- Open http://localhost:8000

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
   - Model status is checked automatically on page load
   - Click "Prepare Training Data" to extract Mad Hatter dialogue
   - Click "Create Trained Model" to create the modelfile
   - Run `./create-model.sh` from host to create the trained model
   - Use "Model Comparison" to compare base vs trained models with suggested prompts
   - View token usage, similarity metrics (ROUGE, BLEU, Jaccard), and detailed insights

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
- Uses base Ollama model (llama3.1) as foundation
- Applies character-specific behavior via system prompt

**Step 3: Training Result**
- New Ollama model "mad-hatter" created and available in Ollama
- Both base and trained models are Ollama models
- Trained model responds in character style when queried
- Shows before/after difference vs base model
- Demonstrates how training changes model behavior

**Training Approach:**
- Uses Ollama Modelfile (system prompt method)
- Simplified for demo (not full parameter fine-tuning)
- Fast to create, easy to understand
- Shows concept of model customization

## Files

**Core Files:**
- `server.py`: Web server, agent logic, training API (Docker-only setup)
- `server-metal.py`: Extended server with Metal GPU support and native Ollama detection
- `training.py`: Model creation and training functions
- `data_processor.py`: Extract character dialogue from text
- `docker-compose.yml`: Container orchestration
- `Dockerfile`: Web service container
- `templates/index.html`: Web UI with model comparison, token usage, and similarity metrics

**Scripts:**
- `run.sh`: Quick start script (handles errors gracefully, pauses before exit)
- `create-model.sh`: Create trained model in Docker Ollama (run from host)
- `cleanup.sh`: Clean up containers and volumes
- `server-metal.py`: Alternative server with Metal GPU support (optional)

**Data:**
- `alice_in_wonderland.txt`: Source text for training
- `training_data/`: Extracted training examples (JSONL format)
- `checkpoints/`: Generated modelfiles
- `outputs/`: Agent-generated files

## Model Comparison Features

The web UI provides comprehensive comparison between base and trained models:

**Suggested Prompts:**
- "What time is it?" - Highlights time obsession
- "Why is a raven like a writing-desk?" - Classic Mad Hatter riddle
- "Which is better math or tea?" - Character preference
- "Tell me about yourself" - Character introduction
- "What's the difference between saying what you mean and meaning what you say?" - Philosophical question
- "Tell me about tea parties" - Tea party theme

**Evaluation Metrics:**
- **Token Usage**: Prompt tokens, completion tokens, total tokens, generation time
- **Standard NLP Metrics**:
  - **[ROUGE-1](https://en.wikipedia.org/wiki/ROUGE_(metric))**: Measures unigram (single word) overlap between responses. Higher scores indicate more shared words. This metric focuses on recall - how many words from the base model's response appear in the trained model's response.
  - **[ROUGE-2](https://en.wikipedia.org/wiki/ROUGE_(metric))**: Measures bigram (2-word phrase) overlap. Higher scores indicate more shared phrases and word pairs. This captures phrase-level similarity, showing if the models use similar word combinations.
  - **[ROUGE-L](https://en.wikipedia.org/wiki/ROUGE_(metric))**: Measures longest common subsequence, capturing sentence structure and word order similarity. This metric evaluates how similar the overall sentence structure is, regardless of exact word matches.
  - **[BLEU](https://en.wikipedia.org/wiki/BLEU)**: Measures n-gram precision with brevity penalty. Standard metric for translation quality evaluation. BLEU focuses on precision - how many n-grams in the trained model's response appear in the base model's response, penalizing overly short responses.
  - **[Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)**: Measures word set overlap using intersection over union. Shows overall vocabulary similarity between responses. Calculates the ratio of shared words to total unique words, providing a balanced view of vocabulary overlap.
- **Insights**: Automatic interpretation of similarity scores and character trait detection

## Training Data

Training data is extracted from:
- Chapter VII: A Mad Tea-Party (primary source)
- Other chapters with Hatter dialogue
- Saved to `training_data/mad_hatter_training.jsonl`

## Model Comparison

Both models are Ollama models:
- **Base model (llama3.1)**: Standard Ollama model, responds normally
- **Trained model (mad-hatter)**: Custom Ollama model created via Modelfile, should:
  - Reference time/tea time obsessively
  - Speak in absurd, nonsensical ways
  - Ask riddles
  - Show philosophical but illogical reasoning

**Comparison Features:**
- Responses displayed first, followed by detailed statistics
- Token consumption tracking (prompt, completion, total, generation time)
- Standard NLP evaluation metrics ([ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)), [BLEU](https://en.wikipedia.org/wiki/BLEU), [Jaccard](https://en.wikipedia.org/wiki/Jaccard_index)) with descriptions and links
- Individual metric conclusions based on score values
- Overall similarity interpretation and insights
- Character trait detection (time references, tea references, riddles)

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
- Verify container is running: `docker ps | grep ollama`
- Check logs: `docker logs training-ollama`

## Cleanup

**Stop containers:**
```bash
docker compose down
```

**Remove volumes (deletes models):**
```bash
docker compose down -v
```

**Full cleanup:**
```bash
./cleanup.sh
```

