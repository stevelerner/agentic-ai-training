#!/usr/bin/env python3
"""
Training Demo Server - Demonstrates agentic AI and model training.
"""

import json
import os
import subprocess
from typing import Optional, Dict, Any, List
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
BASE_MODEL = "llama3.1"
TRAINED_MODEL = "mad-hatter"


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def calculate(expression: str) -> Dict:
    """Evaluate a mathematical expression."""
    try:
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def save_file(filename: str, content: str) -> Dict[str, str]:
    """Save content to a file."""
    try:
        os.makedirs("outputs", exist_ok=True)
        with open(f"outputs/{filename}", 'w') as f:
            f.write(content)
        return {"status": "success", "message": f"Saved to outputs/{filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


TOOLS = {
    "calculate": {
        "function": calculate,
        "description": "Evaluate a mathematical expression",
        "parameters": {"expression": "str"}
    },
    "save_file": {
        "function": save_file,
        "description": "Save content to a file",
        "parameters": {"filename": "str", "content": "str"}
    }
}


# ============================================================================
# AGENT CLASS
# ============================================================================

class TrainingAgent:
    """Agent that can use tools and switch between base and trained models."""
    
    def __init__(self, ollama_host: str = OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.model = BASE_MODEL
        self.conversation_history = []
        
    def call_ollama(self, messages: list, model: Optional[str] = None) -> tuple:
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
                timeout=60
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
            return f"Error calling Ollama: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "eval_duration_ns": 0, "total_duration_ns": 0}
    
    def parse_tool_call(self, text: str) -> Optional[tuple]:
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
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool and return the result."""
        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            tool_func = TOOLS[tool_name]["function"]
            result = tool_func(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def run(self, user_query: str, model: Optional[str] = None, max_iterations: int = 5) -> Dict:
        """Main agent loop with ReAct pattern."""
        model = model or self.model
        
        # For trained character models, don't send system message - let model's system prompt work
        # For base model, use full agent prompt
        if model == TRAINED_MODEL:
            # For character model, add tool instructions to user message instead of system
            # This preserves the model's character-defining system prompt
            enhanced_query = f"""You have access to these tools:
- calculate(expression): Evaluate math expressions  
- save_file(filename, content): Save content to file

When you need to use a tool, respond with ONLY this JSON format:
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}

When you have enough information, provide your final answer without JSON.

User query: {user_query}"""
            messages = [
                {"role": "user", "content": enhanced_query}
            ]
        else:
            # Full agent prompt for base model
            system_prompt = """You are a helpful AI agent with access to tools.

Available tools:
- calculate(expression): Evaluate math expressions
- save_file(filename, content): Save content to file

When you need to use a tool, respond with ONLY this JSON format:
{"tool": "tool_name", "arguments": {"arg1": "value1"}}

When you have enough information, provide your final answer without JSON.

Think step by step and use tools when needed."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        
        steps = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "eval_duration_ns": 0, "total_duration_ns": 0}
        
        for iteration in range(1, max_iterations + 1):
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

def analyze_response(response_text: str) -> Dict[str, Any]:
    """Analyze response for character-specific traits and metrics."""
    text_lower = response_text.lower()
    
    # Mad Hatter character indicators
    character_keywords = {
        "time": ["time", "o'clock", "clock", "hour", "minute", "tea time"],
        "tea": ["tea", "tea party", "cup", "saucer"],
        "riddle": ["riddle", "why is a raven", "writing-desk", "puzzle"],
        "absurd": ["nonsense", "curious", "mad", "wonderland", "alice"],
        "philosophical": ["meaning", "say what you mean", "mean what you say", "philosophy"]
    }
    
    detected_traits = {}
    for trait, keywords in character_keywords.items():
        count = sum(1 for keyword in keywords if keyword in text_lower)
        detected_traits[trait] = count
    
    # Response metrics
    word_count = len(response_text.split())
    char_count = len(response_text)
    sentence_count = response_text.count('.') + response_text.count('!') + response_text.count('?')
    
    # Calculate character score (how much it matches Mad Hatter traits)
    trait_score = sum(detected_traits.values())
    max_possible = sum(len(keywords) for keywords in character_keywords.values())
    character_score = (trait_score / max_possible * 100) if max_possible > 0 else 0
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "detected_traits": detected_traits,
        "character_score": round(character_score, 1),
        "has_time_reference": detected_traits.get("time", 0) > 0,
        "has_tea_reference": detected_traits.get("tea", 0) > 0,
        "has_riddle": detected_traits.get("riddle", 0) > 0
    }


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def check_models() -> Dict[str, bool]:
    """Check which models are available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Strip :latest tag for comparison (Ollama returns names like "llama3.1:latest")
            model_names_no_tag = [name.split(":")[0] for name in model_names]
            return {
                "base": BASE_MODEL in model_names_no_tag,
                "trained": TRAINED_MODEL in model_names_no_tag,
                "all_models": model_names
            }
    except:
        pass
    return {"base": False, "trained": False, "all_models": []}


def prepare_training_data() -> Dict:
    """Extract training data from Alice in Wonderland."""
    try:
        from data_processor import extract_mad_hatter_dialogue
        
        with open('alice_in_wonderland.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]
        
        examples = extract_mad_hatter_dialogue(text)
        
        os.makedirs("training_data", exist_ok=True)
        from data_processor import save_training_data
        save_training_data(examples, "training_data/mad_hatter_training.jsonl")
        
        return {
            "status": "success",
            "examples": len(examples),
            "file": "training_data/mad_hatter_training.jsonl"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def create_trained_model() -> Dict:
    """Create the trained model modelfile. Model creation requires docker command from host."""
    try:
        from training import create_trained_model_via_api
        
        training_data_path = "training_data/mad_hatter_training.jsonl"
        if not os.path.exists(training_data_path):
            prep_result = prepare_training_data()
            if prep_result.get("status") != "success":
                return prep_result
        
        modelfile_path = create_trained_model_via_api(
            base_model=BASE_MODEL,
            training_data_path=training_data_path,
            output_model_name=TRAINED_MODEL
        )
        
        # Modelfile is in checkpoints/ which is mounted as volume
        # Path inside container: /app/checkpoints/mad-hatter.modelfile
        # Path on host: ./checkpoints/mad-hatter.modelfile (relative to project root)
        
        # Check if model already exists
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if TRAINED_MODEL in model_names:
                    return {
                        "status": "success",
                        "message": f"Model '{TRAINED_MODEL}' already exists",
                        "modelfile": modelfile_path
                    }
        except:
            pass
        
        # Docker Ollama - use docker commands
        # Read modelfile content
        with open(modelfile_path, 'r') as f:
            modelfile_content = f.read()
        
        # Try to execute docker commands to create the model
        try:
            # Write modelfile directly to Ollama container using docker exec
            # This avoids path issues with docker cp from inside container
            write_result = subprocess.run([
                "docker", "exec", "-i", "training-ollama",
                "sh", "-c", f"cat > /root/.ollama/{TRAINED_MODEL}.modelfile"
            ], input=modelfile_content, capture_output=True, text=True, timeout=30)
            
            if write_result.returncode != 0:
                raise Exception(f"Failed to write modelfile: {write_result.stderr}")
            
            # Create model in Ollama
            create_result = subprocess.run([
                "docker", "exec", "training-ollama",
                "ollama", "create", TRAINED_MODEL,
                "-f", f"/root/.ollama/{TRAINED_MODEL}.modelfile"
            ], capture_output=True, text=True, timeout=120)
            
            if create_result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Model '{TRAINED_MODEL}' created successfully!",
                    "modelfile": modelfile_path,
                    "output": create_result.stdout
                }
            else:
                # Docker commands failed, provide manual instructions
                host_modelfile_path = f"checkpoints/{TRAINED_MODEL}.modelfile"
                return {
                    "status": "ready",
                    "message": f"Modelfile created. Docker commands failed. Run from host:",
                    "modelfile": modelfile_path,
                    "error": create_result.stderr,
                    "steps": [
                        f"./create-model.sh",
                        f"Or manually: docker cp {host_modelfile_path} training-ollama:/root/.ollama/{TRAINED_MODEL}.modelfile && docker exec training-ollama ollama create {TRAINED_MODEL} -f /root/.ollama/{TRAINED_MODEL}.modelfile"
                    ]
                }
        except FileNotFoundError:
            # Docker command not found - provide manual instructions
            host_modelfile_path = f"checkpoints/{TRAINED_MODEL}.modelfile"
            return {
                "status": "ready",
                "message": f"Modelfile created. Docker not available in container. Run from host:",
                "modelfile": modelfile_path,
                "steps": [
                    f"./create-model.sh",
                    f"Or manually: docker cp {host_modelfile_path} training-ollama:/root/.ollama/{TRAINED_MODEL}.modelfile && docker exec training-ollama ollama create {TRAINED_MODEL} -f /root/.ollama/{TRAINED_MODEL}.modelfile"
                ]
            }
        except Exception as e:
            # Other error - provide manual instructions
            host_modelfile_path = f"checkpoints/{TRAINED_MODEL}.modelfile"
            return {
                "status": "ready",
                "message": f"Modelfile created. Error executing docker: {str(e)}. Run from host:",
                "modelfile": modelfile_path,
                "steps": [
                    f"./create-model.sh",
                    f"Or manually: docker cp {host_modelfile_path} training-ollama:/root/.ollama/{TRAINED_MODEL}.modelfile && docker exec training-ollama ollama create {TRAINED_MODEL} -f /root/.ollama/{TRAINED_MODEL}.modelfile"
                ]
            }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Run agent query with specified model."""
    data = request.json
    query_text = data.get("query", "")
    model = data.get("model", BASE_MODEL)
    
    agent = TrainingAgent()
    result = agent.run(query_text, model=model)
    
    return jsonify(result)


@app.route('/api/models', methods=['GET'])
def models():
    """Get available models."""
    return jsonify(check_models())


@app.route('/api/training/prepare', methods=['POST'])
def prepare_data():
    """Prepare training data from Alice in Wonderland."""
    return jsonify(prepare_training_data())


@app.route('/api/training/create', methods=['POST'])
def create_model():
    """Create trained model."""
    return jsonify(create_trained_model())


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """Compare responses from base and trained models with metrics."""
    data = request.json
    query_text = data.get("query", "")
    
    agent = TrainingAgent()
    
    base_result = agent.run(query_text, model=BASE_MODEL)
    trained_result = agent.run(query_text, model=TRAINED_MODEL)
    
    base_response = base_result.get("final_answer", "")
    trained_response = trained_result.get("final_answer", "")
    
    # Analyze responses
    base_analysis = analyze_response(base_response)
    trained_analysis = analyze_response(trained_response)
    
    # Calculate differences
    token_diff = trained_result.get("usage", {}).get("total_tokens", 0) - base_result.get("usage", {}).get("total_tokens", 0)
    word_diff = trained_analysis["word_count"] - base_analysis["word_count"]
    char_score_diff = trained_analysis["character_score"] - base_analysis["character_score"]
    
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
            "character_score_diff": round(char_score_diff, 1),
            "trained_more_characteristic": char_score_diff > 0
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

