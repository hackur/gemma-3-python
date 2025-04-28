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
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) >= 1
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]

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
    assert "function_call" in data["choices"][0]["message"]
    
def test_streaming_response():
    """Test streaming response capabilities"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemma-3-4b-it",
            "messages": [
                {"role": "user", "content": "Count from 1 to 5"}
            ],
            "stream": True
        }
    )
    assert response.status_code == 200
    
    # Check streaming response format
    for line in response.iter_lines():
        if line:
            chunk = line.decode()
            assert chunk.startswith("data: ")
            
def test_error_handling():
    """Test error handling for invalid requests"""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "invalid-model",
            "messages": [
                {"role": "user", "content": "test"}
            ]
        }
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data