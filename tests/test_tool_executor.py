"""
Tests for the Tool Execution Engine

This module contains unit tests for the Tool Execution Engine, which is responsible
for executing tool calls and managing their lifecycle.
"""

import json
import pytest
import asyncio
from typing import Dict, List, Any

from tool_framework import (
    ToolRegistry, ToolRequest, ToolResponse, ToolResponseStatus,
    ToolError, ToolNotFoundError, ToolValidationError, 
    ToolExecutionError, ToolTimeoutError
)
from tool_executor import ToolExecutor

# Example tool functions for testing

def echo_tool(message: str) -> str:
    """
    Echo the input message
    
    Args:
        message: Message to echo
        
    Returns:
        The same message
    """
    return message

def add_tool(a: int, b: int) -> int:
    """
    Add two numbers
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the two numbers
    """
    return a + b

def error_tool() -> None:
    """
    Always raises an error
    
    Returns:
        Never returns
    """
    raise ValueError("Test error")

def timeout_tool() -> str:
    """
    Sleep for longer than the timeout
    
    Returns:
        Message indicating sleep completed
    """
    import time
    time.sleep(2)  # Sleep for 2 seconds (longer than test timeout)
    return "Sleep completed"

async def async_echo_tool(message: str) -> str:
    """
    Async echo tool
    
    Args:
        message: Message to echo
        
    Returns:
        The same message after a short delay
    """
    await asyncio.sleep(0.1)
    return f"Async: {message}"

# Test fixtures

@pytest.fixture
def tool_registry():
    """Create a tool registry with test tools"""
    registry = ToolRegistry()
    
    # Register test tools
    registry.register_tool(
        name="echo",
        description="Echo the input message",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo"
                }
            },
            "required": ["message"]
        },
        handler_fn=echo_tool
    )
    
    registry.register_tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {
                    "type": "integer",
                    "description": "First number"
                },
                "b": {
                    "type": "integer",
                    "description": "Second number"
                }
            },
            "required": ["a", "b"]
        },
        handler_fn=add_tool
    )
    
    registry.register_tool(
        name="error",
        description="Always raises an error",
        parameters={
            "type": "object",
            "properties": {}
        },
        handler_fn=error_tool
    )
    
    registry.register_tool(
        name="timeout",
        description="Sleep for longer than the timeout",
        parameters={
            "type": "object",
            "properties": {}
        },
        handler_fn=timeout_tool
    )
    
    registry.register_tool(
        name="async_echo",
        description="Async echo tool",
        parameters={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo"
                }
            },
            "required": ["message"]
        },
        handler_fn=async_echo_tool,
        is_async=True
    )
    
    return registry

@pytest.fixture
def tool_executor(tool_registry):
    """Create a tool executor with a short default timeout"""
    return ToolExecutor(tool_registry, default_timeout=1.0)

# Tests

@pytest.mark.asyncio
async def test_execute_tool_success(tool_executor):
    """Test executing a tool successfully"""
    tool_request = ToolRequest(
        name="echo",
        arguments={"message": "Hello, world!"},
        tool_call_id="test-call-1"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.SUCCESS
    assert response.content == "Hello, world!"
    assert response.tool_call_id == "test-call-1"
    assert "execution_time" in response.metadata

@pytest.mark.asyncio
async def test_execute_tool_with_validation(tool_executor):
    """Test executing a tool with argument validation"""
    # Valid arguments
    tool_request = ToolRequest(
        name="add",
        arguments={"a": 5, "b": 3},
        tool_call_id="test-call-2"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.SUCCESS
    assert response.content == "8"
    
    # Invalid arguments (missing required)
    tool_request = ToolRequest(
        name="add",
        arguments={"a": 5},
        tool_call_id="test-call-3"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.ERROR
    assert "Invalid arguments" in response.content
    assert response.metadata["error_type"] == "validation_error"
    
    # Invalid arguments (wrong type)
    tool_request = ToolRequest(
        name="add",
        arguments={"a": "five", "b": 3},
        tool_call_id="test-call-4"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.ERROR
    assert "Invalid arguments" in response.content
    assert response.metadata["error_type"] == "validation_error"

@pytest.mark.asyncio
async def test_execute_tool_not_found(tool_executor):
    """Test executing a non-existent tool"""
    tool_request = ToolRequest(
        name="non_existent_tool",
        arguments={},
        tool_call_id="test-call-5"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.ERROR
    assert "Tool 'non_existent_tool' not found" in response.content
    assert response.metadata["error_type"] == "tool_not_found"

@pytest.mark.asyncio
async def test_execute_tool_error(tool_executor):
    """Test executing a tool that raises an error"""
    tool_request = ToolRequest(
        name="error",
        arguments={},
        tool_call_id="test-call-6"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.ERROR
    assert "Test error" in response.content
    assert response.metadata["error_type"] == "execution_error"

@pytest.mark.asyncio
async def test_execute_tool_timeout(tool_executor):
    """Test executing a tool that times out"""
    tool_request = ToolRequest(
        name="timeout",
        arguments={},
        tool_call_id="test-call-7"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.TIMEOUT
    assert "timeout" in response.content.lower()
    assert response.metadata["error_type"] == "timeout"

@pytest.mark.asyncio
async def test_execute_async_tool(tool_executor):
    """Test executing an async tool"""
    tool_request = ToolRequest(
        name="async_echo",
        arguments={"message": "Async test"},
        tool_call_id="test-call-8"
    )
    
    response = await tool_executor.execute_tool(tool_request)
    
    assert response.status == ToolResponseStatus.SUCCESS
    assert response.content == "Async: Async test"

@pytest.mark.asyncio
async def test_execute_multiple_tools(tool_executor):
    """Test executing multiple tools in parallel"""
    tool_requests = [
        ToolRequest(
            name="echo",
            arguments={"message": "First"},
            tool_call_id="multi-call-1"
        ),
        ToolRequest(
            name="add",
            arguments={"a": 10, "b": 20},
            tool_call_id="multi-call-2"
        ),
        ToolRequest(
            name="async_echo",
            arguments={"message": "Third"},
            tool_call_id="multi-call-3"
        )
    ]
    
    responses = await tool_executor.execute_tools(tool_requests)
    
    assert len(responses) == 3
    
    assert responses[0].status == ToolResponseStatus.SUCCESS
    assert responses[0].content == "First"
    assert responses[0].tool_call_id == "multi-call-1"
    
    assert responses[1].status == ToolResponseStatus.SUCCESS
    assert responses[1].content == "30"
    assert responses[1].tool_call_id == "multi-call-2"
    
    assert responses[2].status == ToolResponseStatus.SUCCESS
    assert responses[2].content == "Async: Third"
    assert responses[2].tool_call_id == "multi-call-3"

@pytest.mark.asyncio
async def test_tool_metrics(tool_executor):
    """Test tool execution metrics"""
    # Execute a tool multiple times
    for i in range(5):
        await tool_executor.execute_tool(
            ToolRequest(
                name="echo",
                arguments={"message": f"Test {i}"},
                tool_call_id=f"metrics-call-{i}"
            )
        )
    
    # Execute a tool that errors
    await tool_executor.execute_tool(
        ToolRequest(
            name="error",
            arguments={},
            tool_call_id="metrics-error"
        )
    )
    
    # Get metrics for specific tool
    echo_metrics = tool_executor.get_metrics("echo")
    assert echo_metrics["total_executions"] == 5
    assert echo_metrics["successful_executions"] == 5
    assert echo_metrics["failed_executions"] == 0
    assert echo_metrics["success_rate"] == 1.0
    
    # Get metrics for error tool
    error_metrics = tool_executor.get_metrics("error")
    assert error_metrics["total_executions"] == 1
    assert error_metrics["successful_executions"] == 0
    assert error_metrics["failed_executions"] == 1
    assert error_metrics["success_rate"] == 0.0
    
    # Get all metrics
    all_metrics = tool_executor.get_metrics()
    assert "echo" in all_metrics
    assert "error" in all_metrics
