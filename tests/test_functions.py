"""
Tests for function calling capabilities.
"""

import pytest
import json
import os
from pathlib import Path
from utils.function_handler import FunctionExecutor

@pytest.fixture
def function_executor():
    return FunctionExecutor()

@pytest.fixture
def sample_script_path():
    return Path("examples/sample_script.py")

@pytest.mark.asyncio
async def test_execute_python(function_executor, sample_script_path):
    """Test Python script execution"""
    result = await function_executor.execute_python(
        script_name="sample_script.py",
        arguments="test"
    )
    assert result["status"] == "success"
    assert result["result"]["return_code"] == 0
    
    # Verify script output
    output = json.loads(result["result"]["stdout"])
    assert "received_args" in output
    assert output["received_args"] == ["test"]
    assert output["processed"] is True

@pytest.mark.asyncio
async def test_get_system_info(function_executor):
    """Test system information retrieval"""
    # Test CPU info
    cpu_result = await function_executor.get_system_info(info_type="cpu")
    assert cpu_result["status"] == "success"
    assert "cpu" in cpu_result["result"]
    assert "percent" in cpu_result["result"]["cpu"]
    
    # Test memory info
    mem_result = await function_executor.get_system_info(info_type="memory")
    assert mem_result["status"] == "success"
    assert "memory" in mem_result["result"]
    assert "total" in mem_result["result"]["memory"]
    assert "used" in mem_result["result"]["memory"]
    
    # Test disk info
    disk_result = await function_executor.get_system_info(info_type="disk")
    assert disk_result["status"] == "success"
    assert "disk" in disk_result["result"]
    assert "total" in disk_result["result"]["disk"]
    assert "used" in disk_result["result"]["disk"]
    
    # Test all info
    all_result = await function_executor.get_system_info(info_type="all")
    assert all_result["status"] == "success"
    assert "cpu" in all_result["result"]
    assert "memory" in all_result["result"]
    assert "disk" in all_result["result"]

@pytest.mark.asyncio
async def test_python_execution_error_handling(function_executor):
    """Test error handling for Python script execution"""
    # Test nonexistent script
    result = await function_executor.execute_python(
        script_name="nonexistent.py",
        arguments=""
    )
    assert result["status"] == "error" or "error" in result
    
    # Test script with invalid syntax
    with open("examples/invalid_script.py", "w") as f:
        f.write("this is not valid python")
    
    result = await function_executor.execute_python(
        script_name="invalid_script.py",
        arguments=""
    )
    assert result["status"] == "error" or "error" in result
    
    # Clean up
    os.remove("examples/invalid_script.py")

@pytest.mark.asyncio
async def test_system_info_error_handling(function_executor):
    """Test error handling for system info retrieval"""
    # Test invalid info type
    result = await function_executor.get_system_info(info_type="invalid")
    assert result["status"] == "error" or "error" in result

@pytest.mark.asyncio
async def test_function_timeout(function_executor, sample_script_path):
    """Test function execution timeout"""
    # Create a script that sleeps
    with open("examples/slow_script.py", "w") as f:
        f.write("import time; time.sleep(10)")
    
    result = await function_executor.execute_python(
        script_name="slow_script.py",
        arguments="",
        timeout=1  # 1 second timeout
    )
    assert result["status"] == "error"
    assert "timeout" in str(result.get("error", "")).lower()
    
    # Clean up
    os.remove("examples/slow_script.py")

@pytest.mark.asyncio
async def test_script_argument_handling(function_executor):
    """Test handling of script arguments"""
    # Test with various argument types
    result = await function_executor.execute_python(
        script_name="sample_script.py",
        arguments='--flag1 value1 --flag2 "value with spaces"'
    )
    assert result["status"] == "success"
    
    output = json.loads(result["result"]["stdout"])
    assert len(output["received_args"]) == 4
    assert output["received_args"][0] == "--flag1"
    assert output["received_args"][1] == "value1"
    assert output["received_args"][2] == "--flag2"
    assert output["received_args"][3] == "value with spaces"