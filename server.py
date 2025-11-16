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
        
    def call_ollama(self, messages: list, model: Optional[str] = None) -> str:
        """Call Ollama API to get LLM response."""
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
            return response.json()["message"]["content"]
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
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
        
        for iteration in range(1, max_iterations + 1):
            response = self.call_ollama(messages, model=model)
            
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
                    "response": response
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
                    "response": response
                })
                return {
                    "final_answer": response,
                    "steps": steps,
                    "model": model
                }
        
        return {
            "final_answer": "Maximum iterations reached.",
            "steps": steps,
            "model": model
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


def check_gpu() -> Dict[str, Any]:
    """Check if Metal GPU is being used by Ollama."""
    try:
        import subprocess
        import os
        
        # Check OLLAMA_HOST to determine if using native Ollama
        ollama_host = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
        
        # If OLLAMA_HOST points to host.docker.internal, we're using native Ollama
        if "host.docker.internal" in ollama_host:
            # Native Ollama on macOS - Metal GPU should be available on Apple Silicon
            return {
                "device": "Metal GPU (native Ollama)",
                "gpu_available": True,
                "note": "Using native Ollama via host.docker.internal. On Apple Silicon, this uses Metal GPU."
            }
        
        # Check if native Ollama is accessible (for cases where OLLAMA_HOST isn't set correctly)
        try:
            # Try to check if Docker Ollama container exists
            docker_check = subprocess.run(
                ["docker", "ps", "--filter", "name=training-ollama", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if "training-ollama" not in docker_check.stdout:
                # Docker Ollama not running - might be native Ollama
                # Check if we can reach Ollama (native would be on host)
                try:
                    result = subprocess.run(
                        ["curl", "-s", "http://host.docker.internal:11434/api/tags"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        return {
                            "device": "Metal GPU (native Ollama)",
                            "gpu_available": True,
                            "note": "Native Ollama detected. On Apple Silicon, this uses Metal GPU."
                        }
                except:
                    pass
        except:
            pass
        
        # Check Docker Ollama logs (will always be CPU)
        try:
            result = subprocess.run(
                ["docker", "logs", "training-ollama"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logs = result.stdout + result.stderr
            
            # Look for device information in logs
            if "device=CPU" in logs:
                return {
                    "device": "CPU (Docker container)",
                    "gpu_available": False,
                    "note": "Docker Desktop on macOS does not support GPU passthrough. Use native Ollama for Metal GPU."
                }
            elif "device=GPU" in logs or "device=Metal" in logs:
                return {
                    "device": "GPU/Metal",
                    "gpu_available": True
                }
            else:
                # Check recent model load logs
                recent_logs = logs.split('\n')[-100:]  # Last 100 lines
                for line in recent_logs:
                    if "device=" in line:
                        if "CPU" in line:
                            return {
                                "device": "CPU (Docker container)",
                                "gpu_available": False,
                                "note": "Docker Desktop on macOS does not support GPU passthrough. Use native Ollama for Metal GPU."
                            }
                        elif "GPU" in line or "Metal" in line:
                            return {
                                "device": "GPU/Metal",
                                "gpu_available": True
                            }
        except:
            pass
        
        # Default - can't determine
        return {
            "device": "Unknown (likely CPU)",
            "gpu_available": False,
            "note": "Cannot determine GPU status. If using Docker, Metal GPU is not available. Use native Ollama for Metal GPU support."
        }
    except Exception as e:
        return {
            "device": "Error checking",
            "gpu_available": False,
            "error": str(e)
        }


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
        
        # Check if using native Ollama or Docker Ollama
        ollama_host_env = os.getenv("OLLAMA_HOST", OLLAMA_HOST)
        using_native_ollama = "host.docker.internal" in ollama_host_env
        
        if using_native_ollama:
            # Native Ollama - create model directly
            try:
                # Check if ollama command is available (from host)
                # Since we're in a container, we need to use the modelfile path
                # The modelfile is in checkpoints/ which is mounted as a volume
                host_modelfile_path = f"checkpoints/{TRAINED_MODEL}.modelfile"
                
                # For native Ollama, we can't execute ollama create from inside container
                # Provide instructions to run from host
                return {
                    "status": "ready",
                    "message": f"Modelfile created. Run from host machine to create model in native Ollama:",
                    "modelfile": modelfile_path,
                    "steps": [
                        f"./create-model-native.sh",
                        f"Or manually: ollama create {TRAINED_MODEL} -f {host_modelfile_path}"
                    ]
                }
            except Exception as e:
                host_modelfile_path = f"checkpoints/{TRAINED_MODEL}.modelfile"
                return {
                    "status": "ready",
                    "message": f"Modelfile created. Run from host:",
                    "modelfile": modelfile_path,
                    "error": str(e),
                    "steps": [
                        f"./create-model-native.sh",
                        f"Or manually: ollama create {TRAINED_MODEL} -f {host_modelfile_path}"
                    ]
                }
        else:
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


@app.route('/api/gpu', methods=['GET'])
def gpu():
    """Check GPU/Metal status."""
    return jsonify(check_gpu())


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
    """Compare responses from base and trained models."""
    data = request.json
    query_text = data.get("query", "")
    
    agent = TrainingAgent()
    
    base_result = agent.run(query_text, model=BASE_MODEL)
    trained_result = agent.run(query_text, model=TRAINED_MODEL)
    
    return jsonify({
        "query": query_text,
        "base_model": {
            "model": BASE_MODEL,
            "response": base_result.get("final_answer", "")
        },
        "trained_model": {
            "model": TRAINED_MODEL,
            "response": trained_result.get("final_answer", "")
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

