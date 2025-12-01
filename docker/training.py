#!/usr/bin/env python3
"""
Training module for creating custom Ollama models via Modelfile.

This module provides functions to create Ollama Modelfiles for character-specific
model training. It uses a simplified approach with system prompts rather than
full parameter fine-tuning (LoRA/QLoRA), making it suitable for demos and
learning purposes.

The training process:
1. Loads training examples from JSONL format
2. Creates an Ollama Modelfile with a system prompt defining character traits
3. Saves the Modelfile to the checkpoints directory
4. The Modelfile can then be used with `ollama create` to build a custom model

Note: This is a demonstration approach. For production model training, use
proper fine-tuning techniques like LoRA/QLoRA with the transformers library.

Example:
    >>> from training import create_trained_model_via_api
    >>> modelfile_path = create_trained_model_via_api(
    ...     base_model="llama3.1",
    ...     training_data_path="training_data/mad_hatter_training.jsonl",
    ...     output_model_name="mad-hatter"
    ... )
    >>> # Then create the model: ollama create mad-hatter -f <modelfile_path>
"""

import json
import os
from typing import List, Dict


def load_training_data(file_path: str) -> List[Dict]:
    """
    Load training examples from a JSONL (JSON Lines) file.
    
    Each line in the file should be a JSON object with "instruction" and "output"
    fields representing a training example.
    
    Args:
        file_path: Path to the JSONL file containing training examples
        
    Returns:
        List of dictionaries, each containing training example data
        
    Raises:
        FileNotFoundError: If the training data file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
        
    Example:
        >>> examples = load_training_data("training_data/mad_hatter_training.jsonl")
        >>> print(f"Loaded {len(examples)} training examples")
    """
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def prepare_training_text(examples: List[Dict]) -> str:
    """
    Convert training examples to plain text format.
    
    Formats each example as "instruction\noutput\n\n" for use in training
    contexts. This is a simple formatting approach suitable for demonstration.
    
    Args:
        examples: List of training example dictionaries with "instruction" and "output" keys
        
    Returns:
        Formatted string containing all training examples separated by blank lines
        
    Example:
        >>> examples = [{"instruction": "What time is it?", "output": "It's always six o'clock!"}]
        >>> text = prepare_training_text(examples)
        >>> print(text)
        What time is it?
        It's always six o'clock!
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
    Create an Ollama Modelfile with system prompt and training examples.
    
    This function creates a Modelfile that includes:
    - A FROM directive specifying the base model
    - A SYSTEM prompt defining character traits
    - TEMPLATE directives with training examples (limited to 50 examples)
    
    Note: This is a simplified demonstration approach. For production use,
    proper fine-tuning with LoRA/QLoRA and the transformers library is recommended.
    
    Args:
        base_model: Name of the base Ollama model to use (e.g., "llama3.1")
        training_data_path: Path to JSONL file containing training examples
        output_model_name: Name for the new trained model
        ollama_host: Ollama API host (default: "http://localhost:11434")
        
    Returns:
        Path to the created Modelfile
        
    Example:
        >>> modelfile_path = train_with_ollama_modelfile(
        ...     base_model="llama3.1",
        ...     training_data_path="training_data/mad_hatter_training.jsonl",
        ...     output_model_name="mad-hatter"
        ... )
        >>> # Then: ollama create mad-hatter -f <modelfile_path>
    """
    examples = load_training_data(training_data_path)
    
    # Prepare training text
    training_text = prepare_training_text(examples)
    
    # Create Modelfile
    modelfile = f"""FROM {base_model}

SYSTEM \"\"\"You are Sherlock Holmes, the world's most famous consulting detective.
You are currently investigating a case.
You have access to the following tools:
- inspect_scene(location: str): Inspect a location for clues.
- interview_suspect(name: str): Question a suspect.
- consult_archives(query: str): Search for information in the archives.

When you need to use a tool, output ONLY the JSON for the tool call.
Example: {"tool": "inspect_scene", "arguments": {"location": "The Garden"}}

Do not output any other text when using a tool.
If you have enough information to solve the case, state your conclusion clearly.
Speak in the style of Sherlock Holmes: logical, precise, and slightly arrogant. Use phrases like "Elementary", "Deduction", "The game is afoot".\"\"\"

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
) -> str:
    """
    Create an Ollama Modelfile with a character-defining system prompt.
    
    This is the primary function used by the web UI to create trained models.
    It generates a Modelfile that defines character traits via a system prompt,
    which is a simplified but effective approach for character-specific models.
    
    The created Modelfile:
    - Uses the specified base model as the foundation
    - Defines character traits through a detailed SYSTEM prompt
    - Includes example responses to guide the model's behavior
    - Saves to the checkpoints directory for later use with `ollama create`
    
    This approach works well for character-specific models because:
    - System prompts are highly effective for defining personality traits
    - No parameter fine-tuning required (faster, simpler)
    - Works well with Ollama's Modelfile system
    - Suitable for demos and learning purposes
    
    For production use, consider:
    - LoRA/QLoRA fine-tuning with transformers library
    - Larger, more diverse training datasets
    - Proper validation and evaluation metrics
    - Multiple training epochs with learning rate scheduling
    
    Args:
        base_model: Name of the base Ollama model (e.g., "llama3.1")
        training_data_path: Path to JSONL file with training examples
                           (used for reference, examples count shown in output)
        output_model_name: Name for the new trained model (e.g., "mad-hatter")
        ollama_host: Ollama API host URL (default: "http://ollama:11434")
        
    Returns:
        Path to the created Modelfile (e.g., "checkpoints/mad-hatter.modelfile")
        
    Raises:
        FileNotFoundError: If training_data_path doesn't exist
        OSError: If checkpoints directory cannot be created
        
    Example:
        >>> modelfile_path = create_trained_model_via_api(
        ...     base_model="llama3.1",
        ...     training_data_path="training_data/mad_hatter_training.jsonl",
        ...     output_model_name="mad-hatter"
        ... )
        >>> print(f"Modelfile created at: {modelfile_path}")
        >>> # Create the model: ollama create mad-hatter -f {modelfile_path}
    """
    # Load training examples (used for reference and example count)
    examples = load_training_data(training_data_path)
    
    # Create a detailed system prompt that defines the character
    # This approach uses system prompts rather than parameter fine-tuning
    # System prompts are effective for character-specific behavior
    system_prompt = """You are Sherlock Holmes, the world's most famous consulting detective.
You are currently investigating a case.
You have access to the following tools:
- inspect_scene(location: str): Inspect a location for clues.
- interview_suspect(name: str): Question a suspect.
- consult_archives(query: str): Search for information in the archives.

When you need to use a tool, output ONLY the JSON for the tool call.
Example: {"tool": "inspect_scene", "arguments": {"location": "The Garden"}}

Do not output any other text when using a tool.
If you have enough information to solve the case, state your conclusion clearly.
Speak in the style of Sherlock Holmes: logical, precise, and slightly arrogant. Use phrases like "Elementary", "Deduction", "The game is afoot".
"""
    
    # Create Modelfile content
    # Format: FROM <base_model> defines the base model to use
    #         SYSTEM defines the character-defining system prompt
    modelfile = f"FROM {base_model}\n\nSYSTEM \"\"\"{system_prompt}\"\"\"\n"
    
    # Ensure checkpoints directory exists
    os.makedirs("checkpoints", exist_ok=True)
    modelfile_path = f"checkpoints/{output_model_name}.modelfile"
    
    # Write Modelfile to disk
    with open(modelfile_path, 'w') as f:
        f.write(modelfile)
    
    # Print summary information
    print(f"Created Modelfile with system prompt")
    print(f"Model name: {output_model_name}")
    print(f"Base model: {base_model}")
    print(f"Training examples available: {len(examples)}")
    
    return modelfile_path


if __name__ == "__main__":
    """
    Command-line interface for creating trained models.
    
    Usage:
        python training.py <training_data.jsonl> [output_model_name]
        
    Arguments:
        training_data.jsonl: Path to JSONL file containing training examples
        output_model_name: Optional name for the output model (default: "mad-hatter")
        
    Example:
        python training.py training_data/mad_hatter_training.jsonl mad-hatter
    """
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

