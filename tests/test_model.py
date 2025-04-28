"""
Tests for model behavior and inference.
"""

import pytest
from llama_cpp import Llama
from pathlib import Path
import json
import os

MODEL_PATH = "gemma-3-4b-it-q4_0/gemma-3-4b-it-q4_0.gguf"

@pytest.fixture
def model():
    """Initialize model for testing"""
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_threads=4
    )

def test_model_loading(model):
    """Test model initialization"""
    assert model is not None
    assert model.ctx is not None

def test_basic_inference(model):
    """Test basic text generation"""
    prompt = "Write a test greeting"
    result = model.create_completion(prompt)
    
    assert result is not None
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "text" in result["choices"][0]
    assert len(result["choices"][0]["text"]) > 0

def test_function_calling_format(model):
    """Test function calling format parsing"""
    prompt = """You are a helpful AI assistant. Use the following function if needed:
    {
        "name": "test_function",
        "description": "A test function",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string"
                }
            },
            "required": ["param1"]
        }
    }
    
    User: Call the test function with 'test' as param1
    Assistant:"""
    
    result = model.create_completion(prompt)
    
    assert result is not None
    response_text = result["choices"][0]["text"]
    
    try:
        response_json = json.loads(response_text)
        assert "name" in response_json
        assert response_json["name"] == "test_function"
        assert "arguments" in response_json
        assert "param1" in response_json["arguments"]
        assert response_json["arguments"]["param1"] == "test"
    except json.JSONDecodeError:
        pytest.fail("Response is not valid JSON")

def test_context_window(model):
    """Test handling of context window limits"""
    # Create a long prompt near context window size
    long_prompt = "test " * 2000
    
    result = model.create_completion(long_prompt, max_tokens=100)
    assert result is not None
    assert "choices" in result
    
def test_error_handling(model):
    """Test model error handling"""
    with pytest.raises(Exception):
        # Invalid parameters should raise an error
        model.create_completion("test", temperature=2.0)

def test_batch_inference(model):
    """Test batch inference capabilities"""
    prompts = [
        "Write a greeting",
        "Count to 3",
        "What is the capital of France?"
    ]
    
    for prompt in prompts:
        result = model.create_completion(prompt)
        assert result is not None
        assert "choices" in result
        assert len(result["choices"]) > 0

def test_metal_acceleration():
    """Test Metal acceleration on Apple Silicon"""
    # Only run on macOS with Apple Silicon
    if os.uname().sysname == "Darwin" and os.uname().machine == "arm64":
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_threads=4,
            n_gpu_layers=1  # Enable GPU acceleration
        )
        
        # Run inference
        result = model.create_completion("test metal acceleration")
        assert result is not None
        
def test_conversation_memory(model):
    """Test conversation memory and context handling"""
    conversation = [
        {"role": "user", "content": "My name is Alice"},
        {"role": "assistant", "content": "Hello Alice!"},
        {"role": "user", "content": "What's my name?"}
    ]
    
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    result = model.create_completion(prompt)
    
    assert result is not None
    assert "Alice" in result["choices"][0]["text"].lower()