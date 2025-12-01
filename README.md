# AI Training Demo
 
A minimal demonstration of agentic AI and model training using Alice in Wonderland.
 
## Versions
 
This repository contains two versions of the demo:
 
1.  **Native Metal Version (`/native-ollama-metal`)**:
    -   **Training**: **True Fine-Tuning** (LoRA via MLX)
    -   **Infrastructure**: Native macOS (Metal GPU inference)
    -   **Best for**: Real-world training demonstration on Apple Silicon.  

2.  **Docker Version (For illustration purposes only) (`/docker`)**:
    -   **Training**: Simulated (System Prompt Engineering via Modelfile)
    -   **Infrastructure**: Docker Containers (CPU inference)
    -   **Best for**: Understanding the concepts without hardware requirements.
    -   Docker version cannot access the GPU, so it is not recommended for production use- example is at end of this doc

## What This Demonstrates
 
**Agentic AI**
- ReAct pattern (Reason → Act → Observe loop)
- Tool calling (calculate)
- Model switching (base vs trained)
 
**Model Training**
- Data extraction from text
- Character-specific fine-tuning
- Before/after model comparison
 
## Quick Start
 
### Native Metal For Mac
 
See [native-ollama-metal/README.md](native-ollama-metal/README.md) for full instructions.
 
```bash
cd native-ollama-metal
./run.sh
```
 
## Agentic AI vs LLM Chat

**LLM Chat**
- Single request → single response
- No actions beyond text generation
- Cannot interact with external systems
- Example: "What time is it?" → "It's currently 3:45 PM"

**Agentic AI (This Demo)**
- Multi-step reasoning loop (ReAct pattern)
- Decides when to use tools autonomously
- Executes actions (calculate)
- Observes results and adapts
- Example: "Ask the Mad Hatter what time it is, then calculate 6 o'clock in minutes" → Queries trained model → Gets "It's always six o'clock!" → Calculates 6 * 60 = 360 → Provides answer

**Key Difference**
- Agentic: Can take actions, use tools, iterate until task complete
- LLM Chat: Only generates text responses

## How Training Works
 
### Docker Version (Simulated Training)
 
**Step 1: Data Extraction**
- Scans Alice in Wonderland text for Mad Hatter dialogue
- Extracts quotes attributed to "Hatter" character
- Formats as instruction-response pairs
 
**Step 2: Model Creation**
- Creates Ollama Modelfile with system prompt
- Defines Mad Hatter character traits (time-obsessed, absurd, riddles)
- Uses base Ollama model (llama3.1) as foundation
- Applies character-specific behavior via system prompt
 
**Step 3: Result**
- New Ollama model "mad-hatter" created
- Demonstrates "Prompt Engineering" as a form of training
 
### Native Metal Version (True Fine-Tuning)
 
**Step 1: Data Preparation**
- Extracts dialogue and formats it into Llama 3 ChatML format
- Creates training and validation datasets (JSONL)
 
**Step 2: LoRA Fine-Tuning**
- Uses Apple's **MLX** framework to train a Low-Rank Adapter (LoRA)
- Updates model weights based on the training data
- Runs for ~100 iterations on the GPU
 
**Step 3: Result**
- A true fine-tuned adapter that changes the model's internal weights
- Can be loaded on top of the base model for inference
 
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
 
## Limitations
 
**Docker Version:**
- Uses system prompt approach (not full fine-tuning)
- Limited training data (single character, single book)
- Modelfile approach is basic
 
**Native Metal Version:**
- Requires Apple Silicon Mac
- Fine-tuning is limited to LoRA (adapters), not full parameter training (for speed/memory reasons)
 

For production training, use:
- Larger datasets
- Proper validation pipelines
- Cloud GPUs for full parameter tuning if needed

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

## Docker Version (Legacy/Illustration)

**Note**: The Docker version is for illustration purposes only and uses simulated training (system prompting). It does not support GPU acceleration on macOS.

**Prerequisites:**
- Docker Desktop running
- 8GB+ RAM available

**Quick Start:**
```bash
cd docker
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
cd docker
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