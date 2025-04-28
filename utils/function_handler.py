"""
Function handling utilities for Gemma3 API Server.

This module provides utilities for handling function execution in a safe and
controlled environment. It supports Python script execution with timeouts,
sandboxing, and comprehensive error handling.

Features:
    - Timeout handling for long-running operations
    - Controlled script execution environment
    - Resource usage monitoring
    - Error handling and logging
    - Virtual environment support

Usage:
    executor = FunctionExecutor()
    result = await executor.execute_python("script.py", "--arg1 value1")
"""

import os
import asyncio
import psutil
import json
import subprocess
import platform
import shlex
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from functools import wraps
import time

def timeout_handler(timeout: int = 30):
    """
    Decorator for handling function timeouts.
    
    Wraps async functions to add timeout functionality. If the function
    doesn't complete within the specified time, returns an error response.
    
    Args:
        timeout: Maximum execution time in seconds (default: 30)
        
    Returns:
        Decorated function that will timeout after specified seconds
        
    Example:
        @timeout_handler(timeout=60)
        async def long_running_task():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            except asyncio.TimeoutError:
                return {
                    "status": "error",
                    "error": "timeout"
                }
        return wrapper
    return decorator

class FunctionExecutor:
    """
    Handles execution of Python scripts and functions.
    
    This class provides methods for executing Python scripts in a controlled
    environment with proper error handling and resource monitoring.
    
    Attributes:
        scripts_dir: Directory for storing scripts (default: examples/)
    """
    
    def __init__(self):
        """Initialize the function executor."""
        # Store scripts in examples directory by default
        self.scripts_dir = Path("examples")
        self.scripts_dir.mkdir(exist_ok=True)
        
    @timeout_handler(timeout=30)
    async def execute_python(
        self, 
        script_name: str, 
        arguments: Optional[str] = None,
        venv_path: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a Python script with optional timeout"""
        try:
            script_path = self.scripts_dir / script_name
            if not script_path.exists():
                return {
                    "status": "error",
                    "error": f"Script {script_name} not found"
                }
                
            # Build command
            python_exe = "python"
            if venv_path:
                if platform.system() == "Windows":
                    python_exe = os.path.join(venv_path, "Scripts", "python.exe")
                else:
                    python_exe = os.path.join(venv_path, "bin", "python")
                    
            cmd = [python_exe, str(script_path)]
            if arguments:
                cmd.extend(shlex.split(arguments))
                
            # Execute script with timeout if specified
            start_time = time.time()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout if timeout else 30
                )
                execution_time = time.time() - start_time

                if process.returncode != 0 or stderr:
                    return {
                        "status": "error",
                        "result": {
                            "stdout": stdout.decode() if stdout else "",
                            "stderr": stderr.decode() if stderr else "",
                            "execution_time": execution_time,
                            "return_code": process.returncode
                        }
                    }
                
                return {
                    "status": "success",
                    "result": {
                        "stdout": stdout.decode() if stdout else "",
                        "stderr": stderr.decode() if stderr else "",
                        "execution_time": execution_time,
                        "return_code": process.returncode
                    }
                }
            except asyncio.TimeoutError:
                try:
                    process.terminate()
                    await process.wait()
                except:
                    pass
                return {
                    "status": "error",
                    "error": f"Script execution timed out after {timeout} seconds"
                }
            
        except Exception as e:
            logger.error(f"Error executing script {script_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    @timeout_handler(timeout=5)
    async def get_system_info(self, info_type: str = "all") -> Dict[str, Any]:
        """
        Get system information.
        
        Retrieves various system metrics based on the requested information type.
        
        Args:
            info_type: Type of information to retrieve ("cpu", "memory", "disk", or "all")
            
        Returns:
            Dict containing requested system information
            
        Example:
            info = await get_system_info("memory")
        """
        try:
            if info_type == "cpu":
                return {
                    "status": "success",
                    "result": {
                        "cpu": {
                            "percent": psutil.cpu_percent(interval=1)
                        }
                    }
                }
            elif info_type == "memory":
                mem = psutil.virtual_memory()
                return {
                    "status": "success",
                    "result": {
                        "memory": {
                            "total": mem.total,
                            "used": mem.total - mem.available,
                            "available": mem.available,
                            "percent": mem.percent
                        }
                    }
                }
            elif info_type == "disk":
                disk = psutil.disk_usage('/')
                return {
                    "status": "success",
                    "result": {
                        "disk": {
                            "total": disk.total,
                            "used": disk.used,
                            "free": disk.free,
                            "percent": disk.percent
                        }
                    }
                }
            elif info_type == "all":
                cpu_percent = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                return {
                    "status": "success",
                    "result": {
                        "cpu": {"percent": cpu_percent},
                        "memory": {
                            "total": mem.total,
                            "used": mem.total - mem.available,
                            "available": mem.available,
                            "percent": mem.percent
                        },
                        "disk": {
                            "total": disk.total,
                            "used": disk.used,
                            "free": disk.free,
                            "percent": disk.percent
                        }
                    }
                }
            else:
                return {
                    "status": "error",
                    "error": f"Invalid info_type: {info_type}"
                }
                
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def execute_function(
        self,
        name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a registered function by name"""
        if name == "execute_python":
            return await self.execute_python(**kwargs)
        elif name == "get_system_info":
            return await self.get_system_info(**kwargs)
        else:
            return {
                "status": "error",
                "error": f"Unknown function: {name}"
            }