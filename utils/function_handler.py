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
    def __init__(self):
        # Store scripts in examples directory by default
        self.scripts_dir = Path("examples")
        self.scripts_dir.mkdir(exist_ok=True)
        
    async def execute_python(
        self,
        script_name: str,
        arguments: str = "",
        timeout: int = 30,
        venv_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a Python script with timeout and virtual environment support"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            return {
                "status": "error",
                "error": f"Script {script_name} not found"
            }
            
        cmd = []
        if venv_path:
            venv_path = Path(venv_path)
            if venv_path.exists():
                activate_script = venv_path / "bin" / "activate"
                cmd.extend(["/bin/bash", "-c", f"source {activate_script} && python {script_path} {arguments}"])
            else:
                return {
                    "status": "error",
                    "error": f"Virtual environment not found at {venv_path}"
                }
        else:
            cmd.extend(["python", str(script_path)])
            if arguments:
                # Use shlex to properly parse quoted arguments
                cmd.extend(shlex.split(arguments))

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
                
                # Check for script execution errors
                if process.returncode != 0:
                    return {
                        "status": "error",
                        "error": stderr.decode() if stderr else "Script execution failed",
                        "stdout": stdout.decode() if stdout else "",
                        "stderr": stderr.decode() if stderr else "",
                        "return_code": process.returncode
                    }
                
                return {
                    "status": "success",
                    "result": {
                        "stdout": stdout.decode() if stdout else "",
                        "stderr": stderr.decode() if stderr else "",
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
                    "error": "timeout"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def get_system_info(self, info_type: str = "all") -> Dict[str, Any]:
        """Get system information based on the requested type"""
        try:
            result = {}
            
            if info_type in ["cpu", "all"]:
                result["cpu"] = {
                    "percent": psutil.cpu_percent(interval=1),
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    "stats": psutil.cpu_stats()._asdict()
                }
                
            if info_type in ["memory", "all"]:
                memory = psutil.virtual_memory()
                result["memory"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                }
                
            if info_type in ["disk", "all"]:
                disk = psutil.disk_usage("/")
                result["disk"] = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
                
            if not result:
                return {
                    "status": "error",
                    "error": f"Invalid info type: {info_type}"
                }
                
            return {
                "status": "success",
                "result": result
            }
            
        except Exception as e:
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