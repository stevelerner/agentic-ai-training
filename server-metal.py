#!/usr/bin/env python3
"""
Training Demo Server with Metal GPU Support - Extends server.py with GPU detection and native Ollama support.
"""

import os
import subprocess
from typing import Dict, Any
from flask import jsonify

# Import everything from server.py
from server import (
    app, OLLAMA_HOST, BASE_MODEL, TRAINED_MODEL,
    TrainingAgent, check_models, prepare_training_data,
    create_trained_model as base_create_trained_model
)


# ============================================================================
# METAL GPU FUNCTIONS
# ============================================================================

def check_gpu() -> Dict[str, Any]:
    """Check if Metal GPU is being used by Ollama."""
    try:
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


def create_trained_model() -> Dict:
    """Create the trained model modelfile with support for both Docker and native Ollama."""
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
        
        # Check if model already exists
        try:
            import requests
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
            # Native Ollama - provide instructions to run from host
            host_modelfile_path = f"checkpoints/{TRAINED_MODEL}.modelfile"
            return {
                "status": "ready",
                "message": f"Modelfile created. Run from host machine to create model in native Ollama:",
                "modelfile": modelfile_path,
                "steps": [
                    f"./create-model-native.sh",
                    f"Or manually: ollama create {TRAINED_MODEL} -f {host_modelfile_path}"
                ]
            }
        else:
            # Docker Ollama - use base function
            return base_create_trained_model()
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# API ENDPOINTS (Metal GPU additions)
# ============================================================================

@app.route('/api/gpu', methods=['GET'])
def gpu():
    """Check GPU/Metal status."""
    return jsonify(check_gpu())


# Override the create model endpoint to use Metal-aware version
@app.route('/api/training/create', methods=['POST'])
def create_model():
    """Create trained model (with native Ollama support)."""
    return jsonify(create_trained_model())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

