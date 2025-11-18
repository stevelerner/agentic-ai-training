#!/usr/bin/env python3
"""
Training module for creating custom Ollama models via Modelfile.
"""

import json
import os
from typing import List, Dict


def load_training_data(file_path: str) -> List[Dict]:
    """Load training examples from a JSONL file."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def create_trained_model_via_api(
    base_model: str,
    training_data_path: str,
    output_model_name: str,
    ollama_host: str = "http://localhost:11434"
) -> str:
    """Create an Ollama Modelfile with a character-defining system prompt."""
    # Load training examples
    examples = load_training_data(training_data_path)
    
    # Create system prompt that defines the character
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
    
    # Create Modelfile content
    modelfile = f"FROM {base_model}\n\nSYSTEM \"\"\"{system_prompt}\"\"\"\n"
    
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    modelfile_path = f"checkpoints/{output_model_name}.modelfile"
    
    # Write Modelfile to disk
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    print(f"Created Modelfile with system prompt")
    print(f"  Model name: {output_model_name}")
    print(f"  Base model: {base_model}")
    print(f"  Training examples: {len(examples)}")
    print(f"  Modelfile: {modelfile_path}")
    
    return modelfile_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python training.py <training_data.jsonl> [output_model_name]")
        print("\nExample:")
        print("  python training.py training_data/mad_hatter_training.jsonl mad-hatter")
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
