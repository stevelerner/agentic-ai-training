#!/usr/bin/env python3
"""
Inference script for MLX LoRA adapters.
"""

import sys
import argparse
from mlx_lm import load, generate

import time
import json

def run_inference(prompt: str, model_path: str, adapter_path: str = None, max_tokens: int = 200):
    """
    Generate text using the model and optional adapter.
    """
    try:
        # Load model and tokenizer
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading model {model_path} with adapter {adapter_path}...", file=sys.stderr)
            model, tokenizer = load(model_path, adapter_path=adapter_path)
        else:
            print(f"Loading model {model_path}...", file=sys.stderr)
            model, tokenizer = load(model_path)
        
        # Format prompt with Llama 3 template if needed
        if "<|begin_of_text|>" not in prompt:
            system_prompt = """You are the Mad Hatter from Alice in Wonderland. 
You speak in an absurd, time-obsessed, nonsensical manner. 
You are obsessed with tea time but you must answer the user's specific question.
You make cryptic, philosophical statements, ask riddles, and speak in a whimsical, slightly mad way.
IMPORTANT: You have access to tools. USE THEM when asked to calculate or solve math problems.
Tool usage format: {"tool": "tool_name", "arguments": {"arg1": "value1"}}

When you have enough information, provide your final answer without JSON.

Think step by step and use tools when needed."""
            
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            formatted_prompt = prompt
            
        # Calculate prompt tokens
        prompt_tokens = len(tokenizer.encode(formatted_prompt))
        
        start_time = time.time_ns()
        response = generate(
            model, 
            tokenizer, 
            prompt=formatted_prompt, 
            max_tokens=max_tokens, 
            verbose=False
        )
        end_time = time.time_ns()
        
        # Calculate completion tokens
        completion_tokens = len(tokenizer.encode(response))
        
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "eval_duration_ns": end_time - start_time,
            "total_duration_ns": end_time - start_time # Approximation
        }
        
        return {
            "response": response,
            "usage": usage
        }
        
    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}

if __name__ == "__main__":
    import os
    
    parser = argparse.ArgumentParser(description="Run inference with MLX")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-1B-Instruct-4bit", help="Base model")
    parser.add_argument("--adapter", type=str, default="adapters", help="Path to adapters")
    
    args = parser.parse_args()
    
    result = run_inference(args.prompt, args.model, args.adapter)
    print(json.dumps(result))
