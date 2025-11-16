#!/usr/bin/env python3
"""
Simple fine-tuning using LoRA for character training.
"""

import json
import os
from typing import List, Dict
import requests


def load_training_data(file_path: str) -> List[Dict]:
    """Load training examples from JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def prepare_training_text(examples: List[Dict]) -> str:
    """
    Convert training examples to text format for fine-tuning.
    Simple format: instruction + output
    """
    training_text = ""
    for ex in examples:
        instruction = ex.get("instruction", "")
        output = ex.get("output", "")
        training_text += f"{instruction}\n{output}\n\n"
    return training_text


def train_with_ollama_modelfile(
    base_model: str,
    training_data_path: str,
    output_model_name: str,
    ollama_host: str = "http://localhost:11434"
):
    """
    Create a Modelfile for Ollama fine-tuning.
    This is a simplified approach - in production you'd use proper LoRA/QLoRA.
    """
    examples = load_training_data(training_data_path)
    
    # Prepare training text
    training_text = prepare_training_text(examples)
    
    # Create Modelfile
    modelfile = f"""FROM {base_model}

SYSTEM \"\"\"You are the Mad Hatter from Alice in Wonderland. 
You speak in an absurd, time-obsessed, nonsensical manner. 
You are always at tea time (six o'clock) and make cryptic, 
philosophical statements. You ask riddles and speak in a 
whimsical, slightly mad way.\"\"\"

# Training examples
"""
    
    # Add examples (limit to avoid too large modelfile)
    for ex in examples[:50]:  # Limit examples
        output = ex.get("output", "").replace('"', '\\"')
        modelfile += f'TEMPLATE """{{{{ .System }}}}\\n\\n{{{{ .Prompt }}}}\\n\\n{output}"\n'
    
    # Save Modelfile
    modelfile_path = f"checkpoints/{output_model_name}.modelfile"
    os.makedirs("checkpoints", exist_ok=True)
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    print(f"Created Modelfile: {modelfile_path}")
    print(f"Training examples: {len(examples)}")
    print(f"\nTo create the model, run:")
    print(f"docker exec training-ollama ollama create {output_model_name} -f /root/.ollama/{output_model_name}.modelfile")
    print(f"\nOr manually:")
    print(f"docker cp {modelfile_path} training-ollama:/root/.ollama/{output_model_name}.modelfile")
    print(f"docker exec training-ollama ollama create {output_model_name} -f /root/.ollama/{output_model_name}.modelfile")
    
    return modelfile_path


def create_trained_model_via_api(
    base_model: str,
    training_data_path: str,
    output_model_name: str,
    ollama_host: str = "http://ollama:11434"
):
    """
    Simplified training: create a model with system prompt and examples.
    Note: This is a demo approach. Real fine-tuning requires more setup.
    """
    examples = load_training_data(training_data_path)
    
    # For demo purposes, we'll create a model with a strong system prompt
    # Real fine-tuning would use LoRA/QLoRA with transformers library
    
    system_prompt = """You are the Mad Hatter from Alice in Wonderland. 
You speak in an absurd, time-obsessed, nonsensical manner. 
You are always at tea time (six o'clock) and make cryptic, 
philosophical statements. You ask riddles and speak in a 
whimsical, slightly mad way. 

Key characteristics:
- Obsessed with time ("It's always six o'clock!")
- Asks riddles ("Why is a raven like a writing-desk?")
- Speaks in contradictions and absurdities
- Philosophical but nonsensical
- Whimsical and slightly mad

Example responses:
- "Time? Why, it's always tea time! Six o'clock, you know!"
- "Why is a raven like a writing-desk? I haven't the slightest idea!"
- "We're all mad here. I'm mad. You're mad."
"""
    
    # Create a simple Modelfile
    modelfile = f"FROM {base_model}\n\nSYSTEM \"\"\"{system_prompt}\"\"\"\n"
    
    # Save it
    os.makedirs("checkpoints", exist_ok=True)
    modelfile_path = f"checkpoints/{output_model_name}.modelfile"
    
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    print(f"Created Modelfile with system prompt")
    print(f"Model name: {output_model_name}")
    print(f"Base model: {base_model}")
    print(f"Training examples available: {len(examples)}")
    
    return modelfile_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python training.py <training_data.jsonl> [output_model_name]")
        sys.exit(1)
    
    training_data_path = sys.argv[1]
    output_model_name = sys.argv[2] if len(sys.argv) > 2 else "mad-hatter"
    
    if not os.path.exists(training_data_path):
        print(f"Error: Training data file not found: {training_data_path}")
        sys.exit(1)
    
    create_trained_model_via_api(
        base_model="llama3.1",
        training_data_path=training_data_path,
        output_model_name=output_model_name
    )

