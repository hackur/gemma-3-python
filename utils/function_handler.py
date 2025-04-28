import os
import subprocess
import json
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

class FunctionExecutor:
    def __init__(self, scripts_dir: str = "scripts"):
        self.scripts_dir = Path(scripts_dir)
        self.scripts_dir.mkdir(exist_ok=True)
        
        # Register available functions
        self.available_functions = {
            "execute_python": self._execute_python_script,
            "get_system_info": self._get_system_info,
            # Add more functions here as needed
        }
        
    def get_function_schemas(self) -> list:
        """Return the list of available function schemas"""
        return [
            {
                "name": "execute_python",
                "description": "Execute a Python script with specified arguments",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script_name": {
                            "type": "string",
                            "description": "Name of the Python script to execute"
                        },
                        "arguments": {
                            "type": "string",
                            "description": "Command line arguments for the script"
                        },
                        "venv_path": {
                            "type": "string",
                            "description": "Optional path to virtual environment"
                        }
                    },
                    "required": ["script_name"]
                }
            },
            {
                "name": "get_system_info",
                "description": "Get system information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "info_type": {
                            "type": "string",
                            "enum": ["cpu", "memory", "disk", "all"],
                            "description": "Type of system information to retrieve"
                        }
                    },
                    "required": ["info_type"]
                }
            }
        ]

    async def execute_function(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function by name with given arguments"""
        if function_name not in self.available_functions:
            raise ValueError(f"Function {function_name} not found")
            
        try:
            result = await self.available_functions[function_name](**arguments)
            return {
                "status": "success",
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _execute_python_script(
        self,
        script_name: str,
        arguments: str = "",
        venv_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a Python script with optional virtual environment"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script {script_name} not found")
        
        cmd = []
        if venv_path:
            venv_path = Path(venv_path)
            if venv_path.exists():
                # Construct virtual environment activation command
                activate_script = venv_path / "bin" / "activate"
                cmd.extend(["/bin/bash", "-c", f"source {activate_script} && python {script_path} {arguments}"])
            else:
                raise ValueError(f"Virtual environment not found at {venv_path}")
        else:
            cmd.extend(["python", str(script_path)])
            if arguments:
                cmd.extend(arguments.split())

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": process.returncode
            }
        except Exception as e:
            raise RuntimeError(f"Failed to execute script: {e}")

    async def _get_system_info(self, info_type: str = "all") -> Dict[str, Any]:
        """Get system information based on the requested type"""
        import psutil
        
        info = {}
        if info_type in ["cpu", "all"]:
            info["cpu"] = {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count()
            }
        
        if info_type in ["memory", "all"]:
            memory = psutil.virtual_memory()
            info["memory"] = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            }
            
        if info_type in ["disk", "all"]:
            disk = psutil.disk_usage("/")
            info["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
            
        return info