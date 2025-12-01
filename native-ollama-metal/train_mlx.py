#!/usr/bin/env python3
"""
Train a LoRA adapter for Llama 3 using MLX.
"""

import os
import sys
import json
import argparse
from mlx_lm import load, lora

def train_model(data_path: str, output_adapter_path: str, base_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
    """
    Fine-tune a model using LoRA.
    """
    print(f"Starting training with base model: {base_model}")
    print(f"Data path: {data_path}")
    print(f"Output adapter path: {output_adapter_path}")
    
    # Ensure data files exist
    train_file = os.path.join(data_path, "train.jsonl")
    valid_file = os.path.join(data_path, "valid.jsonl")
    
    if not os.path.exists(train_file):
        print(f"Error: Training file not found at {train_file}")
        return False
        
    # Training configuration
    # We use a small number of iterations for the demo to be fast
    training_args = {
        "model": base_model,
        "train": True,
        "data": data_path, # Directory containing train.jsonl and valid.jsonl
        "adapter_path": output_adapter_path,
        "batch_size": 1,
        "lora_layers": 16,
        "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0},
        "iters": 100, # Fast demo
        "learning_rate": 5e-5,
        "steps_per_eval": 25,
        "val_batches": 5,
        "save_every": 100,
        "max_seq_length": 1024,
        "seed": 42,
    }
    
    # We will use subprocess to call mlx_lm.lora because it's designed as a CLI
    # but we can also import the training function if we want more control.
    # For simplicity and reliability with the installed package, we'll use the CLI wrapper logic
    # or just call the library function if exposed.
    
    # Actually, mlx_lm.lora has a 'train' function but it's not always stable API.
    # Let's try to run it via command line for maximum compatibility with the installed package.
    
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--train",
        "--data", data_path,
        "--adapter-path", output_adapter_path,
        "--batch-size", str(training_args["batch_size"]),
        "--num-layers", str(training_args["lora_layers"]),
        "--iters", str(training_args["iters"]),
        "--learning-rate", str(training_args["learning_rate"]),
        "--steps-per-eval", str(training_args["steps_per_eval"]),
        "--save-every", str(training_args["save_every"]),
        "--seed", str(training_args["seed"])
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    import subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
        
    process.wait()
    
    if process.returncode == 0:
        print(f"\nTraining completed successfully. Adapters saved to {output_adapter_path}")
        return True
    else:
        print(f"\nTraining failed with return code {process.returncode}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama 3 LoRA adapter")
    parser.add_argument("--data", type=str, default="training_data", help="Path to training data directory")
    parser.add_argument("--output", type=str, default="adapters", help="Path to save adapters")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit", help="Base model to use")
    
    args = parser.parse_args()
    
    train_model(args.data, args.output, args.model)
