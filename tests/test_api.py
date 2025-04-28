"""
Tests for Gemma3 API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from gemma3 import app

client = TestClient(app)

def test_list_models():
    """Test GET /v1/models endpoint"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == "gemma-3-4b-it"

def test_chat_completion():
    """Test POST /v1/chat/completions endpoint"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-3-4b-it",
            "messages": [
                {"role": "user", "content": "Write a test greeting"}
            ],
            "temperature": 0.7
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) >= 1
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    assert data["choices"][0]["message"]["role"] == "assistant"

def test_function_calling():
    """Test function calling capabilities"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-3-4b-it",
            "messages": [
                {"role": "user", "content": "What is the current CPU usage?"}
            ],
            "functions": [
                {
                    "name": "get_system_info",
                    "description": "Get system information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "info_type": {
                                "type": "string",
                                "enum": ["cpu", "memory", "disk", "all"]
                            }
                        },
                        "required": ["info_type"]
                    }
                }
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert "message" in data["choices"][0]
    
    # Function call may or may not be present depending on model response
    if "function_call" in data["choices"][0]["message"]:
        function_call = data["choices"][0]["message"]["function_call"]
        assert "name" in function_call
        assert function_call["name"] == "get_system_info"
        assert "arguments" in function_call

def test_streaming_response():
    """Test streaming response capabilities"""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "gemma-3-4b-it",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            "stream": True
        }
    ) as response:
        assert response.status_code == 200
        
        # Process the SSE stream
        for line in response.iter_lines():
            if line:
                # Remove 'data: ' prefix and parse JSON
                if line.startswith(b'data: '):
                    line = line[6:]  # Skip 'data: '
                    if line != b'[DONE]':
                        chunk = json.loads(line)
                        assert "id" in chunk
                        assert "choices" in chunk
                        if chunk["choices"][0].get("delta", {}).get("content"):
                            assert isinstance(chunk["choices"][0]["delta"]["content"], str)

def test_error_handling():
    """Test error handling for invalid requests"""
    # Test with missing required field
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-3-4b-it",
            "messages": []  # Empty messages array should trigger validation error
        }
    )
    assert response.status_code == 422  # Validation error

def test_conversation_context():
    """Test conversation context handling"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-3-4b-it",
            "messages": [
                {"role": "user", "content": "My name is Bob"},
                {"role": "assistant", "content": "Hello Bob!"},
                {"role": "user", "content": "What's my name?"}
            ],
            "temperature": 0.1
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]
    response_text = data["choices"][0]["message"]["content"].lower()
    assert "bob" in response_text