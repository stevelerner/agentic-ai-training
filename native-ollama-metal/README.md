# Sherlock Holmes Agentic AI Demo (Native Metal) 🕵️‍♂️

This is a **True Agentic AI** demo running natively on macOS with Metal GPU acceleration.
It demonstrates how to train a Llama 3 model to adopt a specific persona (**Sherlock Holmes**) AND use tools to solve complex problems (**Mysteries**).

## What is Agentic AI?
Standard LLMs just predict the next word. **Agentic AI** can:
1.  **Reason**: "I need to find clues to solve this crime."
2.  **Act**: Call tools like `inspect_scene` or `interview_suspect`.
3.  **Observe**: Read the tool output ("Found muddy footprints").
4.  **Deduce**: "The footprints belong to the Gardener!"

## Features
- **Synthetic Mystery Generator**: Creates infinite procedural murder mysteries (Crime -> Clues -> Solution).
- **Detective Tools**:
    - `inspect_scene(location)`: Returns physical clues.
    - `interview_suspect(name)`: Returns witness statements (or lies!).
- **True Fine-Tuning**: Uses Apple's **MLX** framework to train a LoRA adapter on `Llama-3.2-1B`.
- **Metal Acceleration**: Runs locally on your Mac's GPU (no cloud required).

## Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- Ollama installed (for base model comparison): `brew install ollama`

## Quick Start

```bash
./run.sh
```

1.  Open **http://localhost:5001**.
2.  **Generate Mysteries**: Click "Generate Mysteries" to create a synthetic dataset of 200+ cases.
3.  **Train Model**: Click "Train Detective Model" to fine-tune Llama 3 on this data (takes ~2 mins).
4.  **Solve a Case**: Ask Sherlock: *"Someone stole the ruby from the Garden! Who did it?"*

## Project Structure
```
native-ollama-metal/
├── server.py                 # Flask server + Agent Logic (ReAct Loop)
├── sherlock_data_processor.py # [NEW] Generates synthetic mysteries
├── train_mlx.py              # MLX LoRA training script
├── inference_mlx.py          # MLX inference script
├── templates/
│   └── index.html            # Detective Theme UI
└── training_data/            # Generated JSONL datasets
```

## How It Works
1.  **Data Gen**: We generate "multi-turn" conversations where Sherlock uses tools to solve a generated crime.
2.  **Training**: We train the model to output specific JSON tool calls (`{"tool": "inspect_scene"...}`) when faced with a mystery.
3.  **Inference**: The server intercepts these JSON outputs, executes the Python function, and feeds the result back to the model.

## Troubleshooting
- **Training is slow?** Ensure you are on Apple Silicon. First run downloads the model (2GB).
- **Ollama errors?** Ensure `ollama serve` is running.
- **Port in use?** Runs on port 5001.
