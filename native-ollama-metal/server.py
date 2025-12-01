#!/usr/bin/env python3
"""
Training Demo Server - Native Metal Version
Demonstrates agentic AI and model training using native Ollama with Metal GPU support.
"""

import json
import os
import subprocess
import sys
from typing import Optional, Dict, Any, List, Tuple
from flask import Flask, render_template, request, jsonify, Response
import requests

app = Flask(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
BASE_MODEL = "llama3.1"
TRAINED_MODEL = "mad-hatter-mlx"
MLX_BASE_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a mathematical expression safely."""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


TOOLS = {
    "calculate": {
        "function": calculate,
        "description": "Evaluate a mathematical expression",
        "parameters": {"expression": "str"}
    }
}


# ============================================================================
# AGENT CLASS
# ============================================================================

class TrainingAgent:
    """Agent implementing the ReAct pattern (Reason -> Act -> Observe)."""
    
    def __init__(self, ollama_host: str = OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.model = BASE_MODEL
        self.conversation_history = []
        
    def call_ollama(self, messages: list, model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Call Ollama API to get LLM response and token usage."""
        model = model or self.model
        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                "eval_duration_ns": data.get("eval_duration", 0),
                "total_duration_ns": data.get("total_duration", 0)
            }
            return content, usage
        except Exception as e:
            return f"Error calling Ollama: {e}", {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "eval_duration_ns": 0, "total_duration_ns": 0
            }
            
    def format_llama3_prompt(self, messages: list) -> str:
        """Format messages into Llama 3 prompt structure."""
        prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    def call_mlx(self, messages: list) -> Tuple[str, Dict[str, Any]]:
        """Call MLX inference script with formatted prompt."""
        try:
            prompt = self.format_llama3_prompt(messages)
            
            cmd = [
                sys.executable, "inference_mlx.py",
                "--prompt", prompt,
                "--model", MLX_BASE_MODEL,
                "--adapter", "adapters"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                try:
                    # Parse JSON output from inference script
                    output_data = json.loads(result.stdout.strip())
                    
                    if "error" in output_data:
                        return f"Error from MLX: {output_data['error']}", {}
                        
                    content = output_data.get("response", "")
                    usage = output_data.get("usage", {
                        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                        "eval_duration_ns": 0, "total_duration_ns": 0
                    })
                    return content, usage
                except json.JSONDecodeError:
                    # Fallback for non-JSON output (e.g. if script failed before printing JSON)
                    print(f"Failed to parse MLX output as JSON: {result.stdout}")
                    return result.stdout.strip(), {}
            else:
                return f"Error calling MLX: {result.stderr}", {}
        except Exception as e:
            return f"Error calling MLX: {str(e)}", {}
    
    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extract tool call from LLM response."""
        text = text.strip()
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            try:
                obj = json.loads(json_str)
                if "tool" in obj and "arguments" in obj:
                    return obj["tool"], obj["arguments"]
            except:
                pass
        return None
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool and return the result."""
        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            tool_func = TOOLS[tool_name]["function"]
            result = tool_func(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def run(self, user_query: str, model: Optional[str] = None, max_iterations: int = 5) -> Dict[str, Any]:
        """Main agent loop implementing the ReAct pattern."""
        model = model or self.model
        
        if model == TRAINED_MODEL:
            system_prompt = """You are the Mad Hatter from Alice in Wonderland. 
You speak in an absurd, time-obsessed, nonsensical manner. 
You are obsessed with tea time but you must answer the user's specific question.
You make cryptic, philosophical statements, ask riddles, and speak in a whimsical, slightly mad way.
IMPORTANT: You have access to tools. USE THEM when asked to calculate or solve math problems.

Available tools:
- calculate(expression): Evaluate math expressions

Tool usage format: {"tool": "tool_name", "arguments": {"arg1": "value1"}}

When you have enough information, provide your final answer without JSON.

Think step by step and use tools when needed."""
        else:
            system_prompt = """You are a helpful AI agent with access to tools.

Available tools:
- calculate(expression): Evaluate math expressions

When you need to use a tool, respond with ONLY this JSON format:
{"tool": "tool_name", "arguments": {"arg1": "value1"}}

When you have enough information, provide your final answer without JSON.

Think step by step and use tools when needed."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        steps = []
        total_usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "eval_duration_ns": 0, "total_duration_ns": 0
        }
        
        for iteration in range(1, max_iterations + 1):
            if model == TRAINED_MODEL:
                response, usage = self.call_mlx(messages)
            else:
                response, usage = self.call_ollama(messages, model=model)
            
            # Accumulate token usage
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)
            total_usage["eval_duration_ns"] += usage.get("eval_duration_ns", 0)
            total_usage["total_duration_ns"] += usage.get("total_duration_ns", 0)
            
            tool_call = self.parse_tool_call(response)
            
            if tool_call:
                tool_name, arguments = tool_call
                result = self.execute_tool(tool_name, arguments)
                
                steps.append({
                    "iteration": iteration,
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "response": response,
                    "usage": usage
                })
                
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": f"Tool result: {json.dumps(result, indent=2)}"
                })
            else:
                steps.append({
                    "iteration": iteration,
                    "type": "final_answer",
                    "response": response,
                    "usage": usage
                })
                return {
                    "final_answer": response,
                    "steps": steps,
                    "model": model,
                    "usage": total_usage
                }
        
        return {
            "final_answer": "Maximum iterations reached.",
            "steps": steps,
            "model": model,
            "usage": total_usage
        }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_ngrams(text: str, n: int) -> set:
    """Extract n-grams from text."""
    words = text.lower().split()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """Calculate ROUGE-L (Longest Common Subsequence) score."""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    m, n = len(ref_words), len(cand_words)
    if m == 0 or n == 0:
        return 0.0
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == cand_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    if m == 0:
        return 0.0
    return lcs_length / m


def calculate_rouge_n(reference: str, candidate: str, n: int) -> float:
    """Calculate ROUGE-N (n-gram overlap) score."""
    ref_ngrams = get_ngrams(reference, n)
    cand_ngrams = get_ngrams(candidate, n)
    
    if len(ref_ngrams) == 0:
        return 0.0
    
    overlap = len(ref_ngrams & cand_ngrams)
    return overlap / len(ref_ngrams)


def calculate_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """Calculate BLEU score (simplified version)."""
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if len(cand_words) == 0:
        return 0.0
    
    bp = min(1.0, len(cand_words) / len(ref_words)) if len(ref_words) > 0 else 0.0
    
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        overlap = len(ref_ngrams & cand_ngrams)
        precisions.append(overlap / len(cand_ngrams))
    
    if all(p > 0 for p in precisions):
        geo_mean = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        geo_mean = 0.0
    
    return bp * geo_mean


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity coefficient."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def analyze_response(response_text: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a response text with standard evaluation metrics."""
    word_count = len(response_text.split())
    char_count = len(response_text)
    sentence_count = response_text.count('.') + response_text.count('!') + response_text.count('?')
    
    result = {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
    }
    
    if reference_text:
        result["rouge_1"] = round(calculate_rouge_n(reference_text, response_text, 1) * 100, 2)
        result["rouge_2"] = round(calculate_rouge_n(reference_text, response_text, 2) * 100, 2)
        result["rouge_l"] = round(calculate_rouge_l(reference_text, response_text) * 100, 2)
        result["bleu"] = round(calculate_bleu(reference_text, response_text) * 100, 2)
        result["jaccard_similarity"] = round(calculate_jaccard_similarity(reference_text, response_text) * 100, 2)
    
    # Character trait detection
    text_lower = response_text.lower()
    character_keywords = {
        "time": ["time", "o'clock", "clock", "hour", "minute", "tea time"],
        "tea": ["tea", "tea party", "cup", "saucer"],
        "riddle": ["riddle", "why is a raven", "writing-desk", "puzzle"],
    }
    
    detected_traits = {}
    for trait, keywords in character_keywords.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        detected_traits[trait] = count
    
    result["detected_traits"] = detected_traits
    result["has_time_reference"] = detected_traits.get("time", 0) > 0
    result["has_tea_reference"] = detected_traits.get("tea", 0) > 0
    result["has_riddle"] = detected_traits.get("riddle", 0) > 0
    
    return result


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def check_models() -> Dict[str, Any]:
    """Check which models are available."""
    try:
        # Check Ollama models
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        ollama_models = []
        if response.status_code == 200:
            models = response.json().get("models", [])
            ollama_models = [m.get("name", "").split(":")[0] for m in models]
            
        # Check MLX adapters
        mlx_trained = os.path.exists("adapters/adapters.safetensors")
        
        return {
            "base": BASE_MODEL in ollama_models,
            "trained": mlx_trained,
            "all_models": ollama_models + ([TRAINED_MODEL] if mlx_trained else []),
            "metal_enabled": True
        }
    except:
        pass
    return {"base": False, "trained": False, "all_models": [], "metal_enabled": False}


def prepare_training_data() -> Dict[str, Any]:
    """Extract Mad Hatter dialogue from Alice in Wonderland text."""
    try:
        # Run data processor
        cmd = [sys.executable, "data_processor.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Data extraction and formatting complete",
                "output": result.stdout
            }
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_trained_model() -> Dict[str, Any]:
    """Create the trained model using MLX."""
    try:
        # This endpoint now just triggers the training process
        # In a real app, we'd use a task queue. Here we'll stream the output.
        return {"status": "started", "message": "Training started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_training_data_sample() -> Dict[str, Any]:
    """Get a sample of the training data."""
    try:
        data_path = "training_data/train.jsonl"
        if not os.path.exists(data_path):
            return {"status": "error", "message": "Training data not found. Please run 'Prepare Training Data' first."}
        
        samples = []
        with open(data_path, 'r') as f:
            # Read first 5 lines
            for i, line in enumerate(f):
                if i >= 5:
                    break
                try:
                    # Parse JSONL line
                    entry = json.loads(line)
                    # Extract the user/assistant parts from the Llama 3 format for display
                    text = entry.get("text", "")
                    
                    # Simple parsing to make it readable
                    user_part = text.split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
                    assistant_part = text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
                    
                    samples.append({
                        "input": user_part,
                        "output": assistant_part
                    })
                except:
                    continue
                    
        return {
            "status": "success", 
            "samples": samples,
            "total_count": sum(1 for _ in open(data_path))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the main web UI page."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Run an agent query with the specified model."""
    data = request.json
    query_text = data.get("query", "")
    model = data.get("model", BASE_MODEL)
    
    agent = TrainingAgent()
    result = agent.run(query_text, model=model)
    
    return jsonify(result)


@app.route('/api/models', methods=['GET'])
def models():
    """Get list of available models."""
    return jsonify(check_models())


@app.route('/api/training/prepare', methods=['POST'])
def prepare_data():
    """Prepare training data."""
    return jsonify(prepare_training_data())


@app.route('/api/training/data', methods=['GET'])
def get_training_data():
    """Get sample of training data."""
    return jsonify(get_training_data_sample())


@app.route('/api/training/create', methods=['POST'])
def create_model():
    """Start training process."""
    # We'll use Server-Sent Events (SSE) to stream training progress
    def generate():
        cmd = [
            sys.executable, "train_mlx.py",
            "--data", "training_data",
            "--output", "adapters",
            "--model", MLX_BASE_MODEL
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            yield f"data: {json.dumps({'log': line.strip()})}\n\n"
            
        process.wait()
        
        if process.returncode == 0:
            yield f"data: {json.dumps({'status': 'complete', 'message': 'Training finished successfully'})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'error', 'message': 'Training failed'})}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare responses from base and trained models."""
    data = request.json
    query_text = data.get("query", "")
    
    agent = TrainingAgent()
    
    base_result = agent.run(query_text, model=BASE_MODEL)
    trained_result = agent.run(query_text, model=TRAINED_MODEL)
    
    base_response = base_result.get("final_answer", "")
    trained_response = trained_result.get("final_answer", "")
    
    base_analysis = analyze_response(base_response)
    trained_analysis = analyze_response(trained_response, reference_text=base_response)
    
    token_diff = trained_result.get("usage", {}).get("total_tokens", 0) - base_result.get("usage", {}).get("total_tokens", 0)
    word_diff = trained_analysis["word_count"] - base_analysis["word_count"]
    
    return jsonify({
        "query": query_text,
        "base_model": {
            "model": BASE_MODEL,
            "response": base_response,
            "usage": base_result.get("usage", {}),
            "analysis": base_analysis
        },
        "trained_model": {
            "model": TRAINED_MODEL,
            "response": trained_response,
            "usage": trained_result.get("usage", {}),
            "analysis": trained_analysis
        },
        "differences": {
            "token_diff": token_diff,
            "word_diff": word_diff,
            "similarity_metrics": {
                "rouge_1": trained_analysis.get("rouge_1", 0),
                "rouge_2": trained_analysis.get("rouge_2", 0),
                "rouge_l": trained_analysis.get("rouge_l", 0),
                "bleu": trained_analysis.get("bleu", 0),
                "jaccard": trained_analysis.get("jaccard_similarity", 0)
            }
        }
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print("="*60)
    print("Training Demo Server - Native Metal Version")
    print("="*60)
    print(f"Ollama Host: {OLLAMA_HOST}")
    print(f"Server: http://localhost:{port}")
    print("="*60)
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
