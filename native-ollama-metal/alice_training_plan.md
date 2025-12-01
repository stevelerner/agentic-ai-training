# Alice in Wonderland "True Training" Demo Plan

This plan outlines the steps to upgrade the current "prompt engineering" demo to a "true fine-tuning" demo using MLX on macOS Metal.

## Goal
Replace the current Modelfile-based "training" (which is just system prompting) with actual LoRA fine-tuning using the Apple MLX framework, while maintaining the Alice in Wonderland / Mad Hatter persona.

## 1. Environment & Dependencies
We need to add MLX libraries to the project.

- **Action**: Update `requirements.txt`
- **Additions**:
  - `mlx>=0.14.0`
  - `mlx-lm>=0.14.0`
  - `transformers`
  - `huggingface-hub`
  - `datasets`

## 2. Data Processing Enhancements
The current `data_processor.py` extracts raw dialogue. We need to format this into a structured dataset suitable for instruction fine-tuning (e.g., ChatML or Llama 3 instruct format).

- **Action**: Update `data_processor.py`
- **Changes**:
  - Format extracted dialogue into `{messages: [{role: "user", content: ...}, {role: "assistant", content: ...}]}` format.
  - Create a valid `train.jsonl` and `valid.jsonl`.
  - Ensure the "user" prompts are varied (e.g., "What time is it?", "Tell me a riddle", "Who are you?").

## 3. MLX Training Implementation
Create a new script to handle the actual fine-tuning process.

- **Action**: Create `train_mlx.py`
- **Features**:
  - Use `mlx.lm.lora` to fine-tune `Llama-3.2-1B-Instruct` (or 3B) for speed and efficiency on most Macs.
  - Implement a simple training loop or wrap `mlx-lm`'s training CLI.
  - Save adapters to a local directory.
  - **Key Configs**:
    - Rank (LoRA): 8 or 16
    - Epochs: ~100-300 iterations (fast demo)
    - Learning Rate: 1e-4 or 2e-5

## 4. Inference & Serving
Since Ollama cannot natively load raw MLX adapters without conversion (which is heavy), we will use `mlx-lm` for serving the trained model.

- **Action**: Create `inference_mlx.py` or update `server.py`
- **Strategy**:
  - **Option A (Simpler)**: `server.py` calls a helper function that uses `mlx_lm.generate` to produce responses for the "Trained Model".
  - **Option B (Robust)**: Spin up a lightweight `mlx-lm-server` on a separate port (e.g., 8081) and have the main Flask app proxy requests to it.
  - **Decision**: Option A is likely sufficient for a single-user demo. We will load the model once (lazy load) or reload on demand.

## 5. UI & Server Integration
Update the existing Flask application to reflect the changes.

- **Action**: Update `server.py`
- **Changes**:
  - **"Train Model" Button**: Now triggers `train_mlx.py` instead of `ollama create`.
  - **Progress Streaming**: Stream training logs (loss, step) to the UI so the user sees "True Training" happening.
  - **Model Comparison**:
    - "Base Model": Calls Ollama (`llama3.1`).
    - "Trained Model": Calls MLX (`Llama-3.2-1B` + Adapters).
  - **Visuals**: Add a graph or log view showing the loss going down.

## 6. Verification
- Verify that the model actually learns the persona (e.g., checks for "tea time", "riddles").
- Ensure performance is acceptable on M1/M2/M3 chips.

## Directory Structure Updates
```
native-ollama-metal/
├── ...
├── train_mlx.py          # [NEW] MLX training script
├── inference_mlx.py      # [NEW] MLX inference helper
├── adapters/             # [NEW] Where LoRA adapters are saved
└── requirements.txt      # [UPDATED]
```
