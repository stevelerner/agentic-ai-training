#!/usr/bin/env python3
"""
Training Demo Server - Demonstrates agentic AI and model training.

This Flask web server provides a complete demonstration of:
1. Agentic AI: An agent that uses tools (calculate) following the ReAct pattern
   (Reason → Act → Observe loop)
2. Model Training: Creating custom Ollama models with character-specific behavior
3. Model Comparison: Comparing base and trained models using standard NLP evaluation metrics

The server implements:
- A TrainingAgent class that can switch between base and trained models
- Tool execution (mathematical calculations, file saving)
- Training data extraction from Alice in Wonderland
- Model creation via Ollama Modelfiles
- Response analysis using ROUGE, BLEU, and Jaccard similarity metrics
- RESTful API endpoints for web UI interaction

Key Features:
- ReAct pattern: Agent reasons about when to use tools, executes them, observes results
- Model switching: Seamlessly switch between base (llama3.1) and trained (mad-hatter) models
- Character preservation: Trained model maintains character traits via system prompts
- Evaluation metrics: Standard NLP metrics for comparing model responses
- Token tracking: Monitor token usage and generation time for each query

Example Usage:
    Start the server:
        python server.py
    
    Or via Docker:
        docker compose up
    
    Then access the web UI at http://localhost:5000
"""

import json
import os
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
BASE_MODEL = "llama3.1"
TRAINED_MODEL = "mad-hatter"


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def calculate(expression: str) -> Dict[str, Any]:
    """
    Evaluate a mathematical expression safely.
    
    This tool allows the agent to perform mathematical calculations. It uses
    a whitelist approach to ensure only safe mathematical operations are allowed.
    
    Args:
        expression: Mathematical expression as a string (e.g., "25 * 4", "10 + 5")
        
    Returns:
        Dictionary with either:
        - {"result": <number>} on success
        - {"error": <error_message>} on failure
        
    Example:
        >>> calculate("25 * 4")
        {"result": 100}
        >>> calculate("10 + 5 - 3")
        {"result": 12}
    """
    try:
        # Whitelist approach: only allow safe mathematical characters
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
    """
    Agent that implements the ReAct pattern (Reason → Act → Observe).
    
    This agent can:
    - Use tools (calculate) to accomplish tasks
    - Switch between base and trained models
    - Maintain conversation context across tool calls
    - Track token usage and generation time
    
    The ReAct pattern works as follows:
    1. Reason: Agent receives a query and decides if tools are needed
    2. Act: Agent calls tools and receives results
    3. Observe: Agent processes tool results and decides next action
    4. Repeat until final answer is reached
    
    For character models (like mad-hatter), the agent embeds tool instructions
    in the user message rather than using a system prompt, preserving the
    model's character-defining system prompt.
    
    Attributes:
        ollama_host: URL of the Ollama API server
        model: Current model being used (default: BASE_MODEL)
        conversation_history: List of conversation messages (for future use)
        
    Example:
        >>> agent = TrainingAgent()
        >>> result = agent.run("Calculate 25 * 4 and save the result")
        >>> print(result["final_answer"])
        "I calculated 25 * 4 = 100 and saved it to outputs/result.txt"
    """
    
    def __init__(self, ollama_host: str = OLLAMA_HOST):
        """
        Initialize the TrainingAgent.
        
        Args:
            ollama_host: URL of the Ollama API server (default: OLLAMA_HOST)
        """
        self.ollama_host = ollama_host
        self.model = BASE_MODEL
        self.conversation_history = []
        
    def call_ollama(self, messages: list, model: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Call Ollama API to get LLM response and token usage.
        
        Sends a chat request to Ollama and returns both the response content
        and token usage statistics (prompt tokens, completion tokens, duration).
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            model: Model name to use (default: self.model)
            
        Returns:
            Tuple of (response_content, usage_dict) where:
            - response_content: The LLM's text response
            - usage_dict: Dictionary with token usage and timing:
                - prompt_tokens: Number of tokens in the prompt
                - completion_tokens: Number of tokens in the completion
                - total_tokens: Total tokens used
                - eval_duration_ns: Time to generate response (nanoseconds)
                - total_duration_ns: Total request time (nanoseconds)
                
        Raises:
            Returns error message string and zero usage dict on failure
        """
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
    
    def parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extract tool call from LLM response.
        
        Parses JSON tool calls from the LLM's response. Expected format:
        {"tool": "tool_name", "arguments": {"arg1": "value1"}}
        
        Args:
            text: The LLM's response text that may contain a tool call
            
        Returns:
            Tuple of (tool_name, arguments_dict) if a valid tool call is found,
            None otherwise
            
        Example:
            >>> agent.parse_tool_call('{"tool": "calculate", "arguments": {"expression": "25*4"}}')
            ("calculate", {"expression": "25*4"})
        """
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
        """
        Execute a tool and return the result.
        
        Looks up the tool in the TOOLS registry and executes it with the
        provided arguments.
        
        Args:
            tool_name: Name of the tool to execute (must be in TOOLS registry)
            arguments: Dictionary of arguments to pass to the tool function
            
        Returns:
            The result from the tool function, or {"error": <message>} if
            the tool is not found or execution fails
            
        Example:
            >>> agent.execute_tool("calculate", {"expression": "10 + 5"})
            {"result": 15}
        """
        if tool_name not in TOOLS:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            tool_func = TOOLS[tool_name]["function"]
            result = tool_func(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def run(self, user_query: str, model: Optional[str] = None, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Main agent loop implementing the ReAct pattern.
        
        This is the core agent method that:
        1. Formats the query with tool instructions (preserving character prompts for trained models)
        2. Calls the LLM to get a response
        3. Parses the response for tool calls
        4. Executes tools if needed and feeds results back to the LLM
        5. Repeats until a final answer is reached or max_iterations is exceeded
        
        For trained character models (like mad-hatter), tool instructions are
        embedded in the user message rather than using a system prompt. This
        preserves the model's character-defining system prompt.
        
        Args:
            user_query: The user's question or request
            model: Model to use (default: self.model, can be BASE_MODEL or TRAINED_MODEL)
            max_iterations: Maximum number of tool-call iterations (default: 5)
            
        Returns:
            Dictionary containing:
            - final_answer: The agent's final response
            - steps: List of step dictionaries showing the agent's reasoning process
            - model: The model used
            - usage: Accumulated token usage across all iterations
            
        Example:
            >>> agent = TrainingAgent()
            >>> result = agent.run("What is 25 times 4?")
            >>> print(result["final_answer"])
            "25 times 4 equals 100."
        """
        model = model or self.model
        
        # For trained character models, don't send system message - let model's system prompt work
        # For base model, use full agent prompt
        if model == TRAINED_MODEL:
            # For character model, add tool instructions to user message instead of system
            # This preserves the model's character-defining system prompt
            enhanced_query = f"""You have access to this tool:
- calculate(expression): Evaluate math expressions

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

def get_ngrams(text: str, n: int) -> set:
    """
    Extract n-grams (contiguous sequences of n words) from text.
    
    Used for calculating ROUGE-N and BLEU scores, which measure n-gram overlap
    between reference and candidate texts.
    
    Args:
        text: Input text to extract n-grams from
        n: Size of n-grams (1=unigrams, 2=bigrams, etc.)
        
    Returns:
        Set of n-gram tuples (each tuple contains n words)
        
    Example:
        >>> get_ngrams("the quick brown fox", 2)
        {('the', 'quick'), ('quick', 'brown'), ('brown', 'fox')}
    """
    words = text.lower().split()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """
    Calculate ROUGE-L (Longest Common Subsequence) score.
    
    ROUGE-L measures the longest common subsequence (LCS) of words between
    reference and candidate texts. It captures sentence-level structure
    similarity and is less sensitive to word order than n-gram metrics.
    
    Score range: 0.0 to 1.0 (higher is better)
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text to evaluate
        
    Returns:
        ROUGE-L score as a float between 0.0 and 1.0
        
    Reference:
        https://aclanthology.org/W04-1013/
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    # LCS using dynamic programming
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
    """
    Calculate ROUGE-N (n-gram overlap) score.
    
    ROUGE-N measures the overlap of n-grams between reference and candidate
    texts. ROUGE-1 measures unigram (word) overlap, ROUGE-2 measures bigram
    overlap, etc. Higher scores indicate more shared n-grams.
    
    Score range: 0.0 to 1.0 (higher is better)
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text to evaluate
        n: Size of n-grams (1 for ROUGE-1, 2 for ROUGE-2, etc.)
        
    Returns:
        ROUGE-N score as a float between 0.0 and 1.0
        
    Reference:
        https://aclanthology.org/W04-1013/
    """
    ref_ngrams = get_ngrams(reference, n)
    cand_ngrams = get_ngrams(candidate, n)
    
    if len(ref_ngrams) == 0:
        return 0.0
    
    overlap = len(ref_ngrams & cand_ngrams)
    return overlap / len(ref_ngrams)


def calculate_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score (simplified version).
    
    BLEU (Bilingual Evaluation Understudy) measures n-gram precision with
    a brevity penalty. It's commonly used for machine translation evaluation.
    This implementation calculates precision for n-grams up to max_n and
    applies a brevity penalty.
    
    Score range: 0.0 to 1.0 (higher is better)
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text to evaluate
        max_n: Maximum n-gram order to consider (default: 4)
        
    Returns:
        BLEU score as a float between 0.0 and 1.0
        
    Reference:
        https://aclanthology.org/P02-1040/
    """
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if len(cand_words) == 0:
        return 0.0
    
    # Brevity penalty
    bp = min(1.0, len(cand_words) / len(ref_words)) if len(ref_words) > 0 else 0.0
    
    # Precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)
        
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        overlap = len(ref_ngrams & cand_ngrams)
        precisions.append(overlap / len(cand_ngrams))
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        geo_mean = 0.0
    
    return bp * geo_mean


def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity coefficient (word overlap).
    
    Jaccard similarity measures the overlap between two sets of words.
    It's calculated as the intersection divided by the union of word sets.
    This provides a simple measure of how many words are shared between texts.
    
    Score range: 0.0 to 1.0 (higher is better)
    
    Args:
        text1: First text to compare
        text2: Second text to compare
        
    Returns:
        Jaccard similarity score as a float between 0.0 and 1.0
        
    Example:
        >>> calculate_jaccard_similarity("the cat sat", "the dog sat")
        0.5  # 2 shared words ("the", "sat") out of 4 unique words total
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def analyze_response(response_text: str, reference_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze a response text with standard evaluation metrics.
    
    Calculates basic text statistics and, if a reference text is provided,
    computes similarity metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Jaccard).
    Also detects character-specific traits (time references, tea references, riddles)
    for the Mad Hatter character.
    
    Args:
        response_text: The text to analyze
        reference_text: Optional reference text for similarity calculations
        
    Returns:
        Dictionary containing:
        - word_count: Number of words
        - char_count: Number of characters
        - sentence_count: Approximate sentence count
        - rouge_1, rouge_2, rouge_l, bleu, jaccard_similarity: If reference provided
        - detected_traits: Dictionary of character trait keyword counts
        - has_time_reference, has_tea_reference, has_riddle: Boolean flags
        
    Example:
        >>> analyze_response("It's always six o'clock!", "It's tea time!")
        {
            "word_count": 4,
            "rouge_1": 25.0,
            "has_time_reference": True,
            ...
        }
    """
    # Basic metrics
    word_count = len(response_text.split())
    char_count = len(response_text)
    sentence_count = response_text.count('.') + response_text.count('!') + response_text.count('?')
    
    result = {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
    }
    
    # If reference text provided, calculate similarity metrics
    if reference_text:
        result["rouge_1"] = round(calculate_rouge_n(reference_text, response_text, 1) * 100, 2)
        result["rouge_2"] = round(calculate_rouge_n(reference_text, response_text, 2) * 100, 2)
        result["rouge_l"] = round(calculate_rouge_l(reference_text, response_text) * 100, 2)
        result["bleu"] = round(calculate_bleu(reference_text, response_text) * 100, 2)
        result["jaccard_similarity"] = round(calculate_jaccard_similarity(reference_text, response_text) * 100, 2)
    
    # Character trait detection (for insights, not scoring)
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
    """
    Check which models are available in Ollama.
    
    Queries the Ollama API to see which models are installed. Used by the
    web UI to display model status and enable/disable features accordingly.
    
    Returns:
        Dictionary containing:
        - base: Boolean indicating if BASE_MODEL (llama3.1) is available
        - trained: Boolean indicating if TRAINED_MODEL (mad-hatter) is available
        - all_models: List of all model names found in Ollama
        
    Example:
        >>> check_models()
        {
            "base": True,
            "trained": True,
            "all_models": ["llama3.1:latest", "mad-hatter:latest"]
        }
    """
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


def prepare_training_data() -> Dict[str, Any]:
    """
    Extract Mad Hatter dialogue from Alice in Wonderland text.
    
    Processes the alice_in_wonderland.txt file to extract dialogue attributed
    to the Mad Hatter character. The extracted examples are saved as JSONL
    format for use in model training.
    
    Returns:
        Dictionary with either:
        - {"status": "success", "examples": <count>, "file": <path>} on success
        - {"status": "error", "message": <error_message>} on failure
        
    Example:
        >>> prepare_training_data()
        {
            "status": "success",
            "examples": 43,
            "file": "training_data/mad_hatter_training.jsonl"
        }
    """
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


def create_trained_model() -> Dict[str, Any]:
    """
    Create the trained model using Ollama Modelfile.
    
    This function:
    1. Ensures training data exists (creates it if needed)
    2. Generates a Modelfile with character-defining system prompt
    3. Attempts to automatically create the model in Docker Ollama
    4. Falls back to manual instructions if Docker commands fail
    
    The Modelfile is saved to checkpoints/ directory and then copied into
    the Ollama container. The model is created using `ollama create`.
    
    Returns:
        Dictionary with status and details:
        - {"status": "success", "message": <msg>, "modelfile": <path>, "output": <stdout>}
        - {"status": "ready", "message": <msg>, "modelfile": <path>, "steps": [<instructions>]}
        - {"status": "error", "message": <error_message>}
        
    Example:
        >>> create_trained_model()
        {
            "status": "success",
            "message": "Model 'mad-hatter' created successfully!",
            "modelfile": "checkpoints/mad-hatter.modelfile"
        }
    """
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
    """
    Serve the main web UI page.
    
    Returns:
        Rendered HTML template (index.html) with the demo interface
    """
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """
    Run an agent query with the specified model.
    
    Accepts a JSON request with:
    - query: The user's question or request
    - model: Optional model name (default: BASE_MODEL)
    
    Returns:
        JSON response with agent execution results:
        - final_answer: The agent's final response
        - steps: List of agent steps (tool calls, responses)
        - model: Model used
        - usage: Token usage statistics
    """
    data = request.json
    query_text = data.get("query", "")
    model = data.get("model", BASE_MODEL)
    
    agent = TrainingAgent()
    result = agent.run(query_text, model=model)
    
    return jsonify(result)


@app.route('/api/models', methods=['GET'])
def models():
    """
    Get list of available models in Ollama.
    
    Returns:
        JSON response with model availability status:
        - base: Boolean for BASE_MODEL availability
        - trained: Boolean for TRAINED_MODEL availability
        - all_models: List of all available model names
    """
    return jsonify(check_models())


@app.route('/api/training/prepare', methods=['POST'])
def prepare_data():
    """
    Prepare training data by extracting Mad Hatter dialogue.
    
    Extracts character dialogue from alice_in_wonderland.txt and saves it
    as JSONL format for model training.
    
    Returns:
        JSON response with extraction results:
        - status: "success" or "error"
        - examples: Number of examples extracted (if successful)
        - file: Path to saved training data file (if successful)
        - message: Error message (if error)
    """
    return jsonify(prepare_training_data())


@app.route('/api/training/create', methods=['POST'])
def create_model():
    """
    Create the trained model using the generated Modelfile.
    
    Generates a Modelfile and attempts to create the model in Ollama.
    May return manual instructions if automatic creation fails.
    
    Returns:
        JSON response with creation status and details (see create_trained_model())
    """
    return jsonify(create_trained_model())


@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    Compare responses from base and trained models with evaluation metrics.
    
    Runs the same query through both models and calculates:
    - Token usage for each model
    - Response analysis (word count, character count, etc.)
    - Similarity metrics (ROUGE-1, ROUGE-2, ROUGE-L, BLEU, Jaccard)
    - Character trait detection
    
    Accepts JSON request with:
    - query: The question/request to compare
    
    Returns:
        JSON response with:
        - query: The original query
        - base_model: Response, usage, and analysis for base model
        - trained_model: Response, usage, and analysis for trained model
        - differences: Token/word differences and similarity metrics
    """
    data = request.json
    query_text = data.get("query", "")
    
    agent = TrainingAgent()
    
    base_result = agent.run(query_text, model=BASE_MODEL)
    trained_result = agent.run(query_text, model=TRAINED_MODEL)
    
    base_response = base_result.get("final_answer", "")
    trained_response = trained_result.get("final_answer", "")
    
    # Analyze responses (using base as reference for similarity metrics)
    base_analysis = analyze_response(base_response)
    trained_analysis = analyze_response(trained_response, reference_text=base_response)
    
    # Calculate differences
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
    """
    Start the Flask development server.
    
    The server runs on all interfaces (0.0.0.0) on port 5000, making it
    accessible from the host machine when running in Docker.
    
    Debug mode is enabled for development (auto-reload on code changes).
    """
    app.run(host="0.0.0.0", port=5000, debug=True)

