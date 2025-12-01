# Sherlock Holmes Agentic AI Demo

A demonstration of **Agentic AI** and **True Fine-Tuning** using the Sherlock Holmes persona.

## Overview

This project demonstrates how to train a small language model (Llama 3.1 1B) to become a functional "Detective Agent". It goes beyond simple chat by teaching the model to:
1.  **Adopt a Persona**: Speak like Sherlock Holmes ("Elementary", "Deduction").
2.  **Use Tools**: Autonomously call tools like `inspect_scene` and `interview_suspect` to gather information.
3.  **Follow the ReAct Pattern**: Reason -> Act -> Observe -> Conclude.

## Key Features

-   **Native Metal Training**: Uses Apple's **MLX** framework for efficient LoRA fine-tuning on Mac GPUs.
-   **Synthetic Data Generation**: Generates hundreds of "Golden Data" mystery examples (Crime -> Clues -> Solution) to teach the model.
-   **Side-by-Side Comparison**: Visually compares the Base Model (verbose, chatty) vs. the Trained Agent (concise, functional).
-   **Educational Insights**: Explains *why* training works (LoRA, Loss Functions, Token Efficiency).

## Quick Start (Mac with Apple Silicon)

1.  **Setup**:
    ```bash
    cd native-ollama-metal
    ./run.sh
    ```

2.  **Open Web UI**:
    Go to `http://localhost:5001`

3.  **Follow the Steps**:
    -   **Step 1: Generate Mysteries**: Creates synthetic training data.
    -   **Step 2: Train Model**: Fine-tunes Llama 3.1 using MLX (takes ~2-3 mins).
    -   **Step 3: Solve Case**: Compare the models on a new mystery.

## Why Training Matters

The demo highlights the difference between a **General LLM** and a **Specialized Agent**:

| Feature | Base Model (Llama 3.1) | Trained Agent (Sherlock) |
| :--- | :--- | :--- |
| **Behavior** | Chatty, polite, verbose | Concise, direct, focused |
| **Tool Use** | Tries to use tools but adds filler text | Strictly follows JSON syntax |
| **Reliability** | Often hits iteration limits (loops) | Solves cases in 1-2 turns |
| **Persona** | Generic AI Assistant | Sherlock Holmes |

## How It Works

### 1. Synthetic Data ("The Textbook")
We don't just train on the book text. We generate **Synthetic Data** that teaches the model *how to behave*.
-   **Input**: "Sherlock, someone stole the ruby!"
-   **Target**: "The game is afoot! {"tool": "inspect_scene", ...}"

### 2. LoRA Fine-Tuning
We use **Low-Rank Adaptation (LoRA)** to train a small "adapter" layer on top of the base model. This allows us to change the model's behavior without retraining the entire 7B parameter network, making it fast and efficient on consumer hardware.

### 3. Verification
The web UI provides real-time metrics:
-   **Token Usage**: See how the trained model becomes more efficient.
-   **Trait Detection**: Tracks usage of "Deduction", "Clue", and "Elementary".
-   **Similarity Metrics**: ROUGE/BLEU scores to measure style drift.

## Project Structure

-   `native-ollama-metal/`: Core project files.
    -   `server.py`: Flask backend and Agent logic.
    -   `sherlock_data_processor.py`: Generates synthetic mysteries.
    -   `train_mlx.py`: MLX training script.
    -   `templates/index.html`: Web UI.

## Requirements

-   macOS with Apple Silicon (M1/M2/M3)
-   Python 3.9+
-   [Ollama](https://ollama.com/) installed and running