"""
Tests for function calling capabilities.
"""

import pytest
import json
import os
from pathlib import Path
import asyncio
from unittest.mock import patch, AsyncMock

from utils.function_handler import FunctionExecutor

@pytest.fixture
def function_executor():
    return FunctionExecutor()

# Determine the project root based on the test file location
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"

@pytest.fixture
def sample_script_path():
    """Fixture that provides path to sample script"""
    return EXAMPLES_DIR / "sample_script.py"

@pytest.mark.asyncio
async def test_execute_python(function_executor, sample_script_path):
    """Test Python script execution"""
    print("Executing test_execute_python")
    try:
        result = await function_executor.execute_python(
            script_name="sample_script.py",
            arguments="--name test --count 5"
        )
        assert result["status"] == "success"
        assert result["result"]["return_code"] == 0

        # Verify script output
        output = json.loads(result["result"]["stdout"])
        assert output["input"]["name"] == "test"
        assert output["input"]["count"] == 5
        assert output["success"] is True
    except Exception as e:
        print(f"Error in test_execute_python: {e}")
        raise e

@pytest.mark.asyncio
async def test_get_system_info(function_executor):
    """Test system information retrieval"""
    print("Executing test_get_system_info")
    try:
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
    except Exception as e:
        print(f"Error in test_get_system_info: {e}")
        raise e


@pytest.mark.asyncio
async def test_python_execution_error_handling(function_executor, tmp_path):
    """Test error handling for Python script execution"""
    print("Executing test_python_execution_error_handling")
    try:
        # Test nonexistent script
        result = await function_executor.execute_python(
            script_name="nonexistent_script.py", # Use a clearly non-existent name
            arguments=""
        )
        assert result["status"] == "error"
        assert result.get("error") == "Script nonexistent_script.py not found"

        # Test script with invalid syntax
        invalid_script = tmp_path / "invalid_syntax.py"
        with open(invalid_script, "w") as f:
            f.write("this is not valid python")

        result = await function_executor.execute_python(
            script_name=str(invalid_script),
            arguments=""
        )
        assert result["status"] == "error"
        assert "invalid syntax" in result.get("error", "").lower()
    except Exception as e:
        print(f"Error in test_python_execution_error_handling: {e}")
        raise e
@pytest.mark.asyncio
async def test_system_info_error_handling(function_executor):
    """Test error handling for system info retrieval"""
    print("Executing test_system_info_error_handling")
    try:
        # Test invalid info type
        result = await function_executor.get_system_info(info_type="invalid")
        assert result["status"] == "error" or "error" in result
        assert "Invalid info_type" in result.get("error", "") # Check for specific error message
    except Exception as e:
        print(f"Error in test_system_info_error_handling: {e}")
        raise e

@pytest.mark.asyncio
async def test_function_timeout_mocked(function_executor):
    """Test function execution timeout handling using mock"""
    # Patch asyncio.wait_for within the function_handler module
    with patch('utils.function_handler.asyncio.wait_for', side_effect=asyncio.TimeoutError) as mock_wait_for:
        # Call execute_python. We need a script that exists to pass the initial check,
        # but the patch will prevent the actual script execution from completing.
        # Using sample_script.py which should exist in the examples dir.
        result = await function_executor.execute_python(
            script_name="sample_script.py",
            arguments=""
        )

        # Assert that the timeout handler caught the error and returned the correct structure
        assert result == {"status": "error", "error": "timeout"}
        # Verify wait_for was called (meaning the function execution started)
        mock_wait_for.assert_called_once()

    # Also test timeout for get_system_info
    with patch('utils.function_handler.asyncio.wait_for', side_effect=asyncio.TimeoutError) as mock_wait_for_sys:
        result_sys = await function_executor.get_system_info(info_type="cpu")
        assert result_sys == {"status": "error", "error": "timeout"}
        mock_wait_for_sys.assert_called_once()

@pytest.mark.asyncio
async def test_script_argument_handling(function_executor):
    """Test handling of script arguments"""
    print("Executing test_script_argument_handling")
    try:
        # Test with various argument types
        # Test with various argument types
        result = await function_executor.execute_python(
            script_name="sample_script.py", # Assuming sample_script.py is in examples
            arguments="--name test --count 5"
        )
        assert result["status"] == "success"

        output = json.loads(result["result"]["stdout"])
        assert output["input"]["name"] == "test"
        assert output["input"]["count"] == 5
        assert output["success"] is True
    except Exception as e:
        print(f"Error in test_script_argument_handling: {e}")
        raise e

# --- New tests will be added below ---
@pytest.mark.asyncio
async def test_execute_function_dispatcher(function_executor):
    """Test the main execute_function dispatcher logic"""
    print("Executing test_execute_function_dispatcher")
    try:
        # Mock the underlying methods
        function_executor.execute_python = AsyncMock(return_value={"status": "mock_python_success"})
        function_executor.get_system_info = AsyncMock(return_value={"status": "mock_sysinfo_success"})

        # Test calling execute_python
        python_args = {"script_name": "test.py", "arguments": "--test"}
        result_python = await function_executor.execute_function(name="execute_python", **python_args)
        assert result_python == {"status": "mock_python_success"}
        function_executor.execute_python.assert_called_once_with(**python_args)
        function_executor.get_system_info.assert_not_called() # Ensure the other wasn't called

        # Reset mocks for the next call
        function_executor.execute_python.reset_mock()
        function_executor.get_system_info.reset_mock()

        # Test calling get_system_info
        sysinfo_args = {"info_type": "cpu"}
        result_sysinfo = await function_executor.execute_function(name="get_system_info", **sysinfo_args)
        assert result_sysinfo == {"status": "mock_sysinfo_success"}
        function_executor.get_system_info.assert_called_once_with(**sysinfo_args)
        function_executor.execute_python.assert_not_called() # Ensure the other wasn't called

        # Reset mocks
        function_executor.execute_python.reset_mock()
        function_executor.get_system_info.reset_mock()

        # Test calling an unknown function
        unknown_args = {"arg1": "value1"}
        result_unknown = await function_executor.execute_function(name="unknown_function", **unknown_args)
        assert result_unknown == {
            "status": "error",
            "error": "Unknown function: unknown_function"
        }
        function_executor.execute_python.assert_not_called()
        function_executor.get_system_info.assert_not_called()
    except Exception as e:
        print(f"Error in test_execute_function_dispatcher: {e}")
        raise e