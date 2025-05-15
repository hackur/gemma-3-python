"""
Tool Execution Engine for Gemma 3 Proxy Server

This module implements an engine to execute tool calls and manage their lifecycle.
It provides support for both synchronous and asynchronous tools, timeout handling,
and result formatting.
"""

import os
import json
import time
import asyncio
import inspect
import logging
import traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Type, Tuple

from tool_framework import (
    ToolRegistry, ToolRequest, ToolResponse, ToolResponseStatus,
    ToolError, ToolNotFoundError, ToolValidationError, 
    ToolExecutionError, ToolTimeoutError, generate_tool_id
)

# Configure logging
logger = logging.getLogger("tool_executor")

class ExecutionMetrics:
    """Metrics for tool execution"""
    
    def __init__(self):
        """Initialize execution metrics"""
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.timed_out_executions = 0
        self.execution_times: List[float] = []
        
    def record_execution(self, success: bool, duration: float, timed_out: bool = False):
        """
        Record metrics for a tool execution
        
        Args:
            success: Whether the execution was successful
            duration: Execution time in seconds
            timed_out: Whether the execution timed out
        """
        self.total_executions += 1
        self.execution_times.append(duration)
        
        if timed_out:
            self.timed_out_executions += 1
        elif success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            
    def get_average_execution_time(self) -> float:
        """Get the average execution time in seconds"""
        if not self.execution_times:
            return 0.0
        return sum(self.execution_times) / len(self.execution_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary"""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "timed_out_executions": self.timed_out_executions,
            "average_execution_time": self.get_average_execution_time(),
            "success_rate": (self.successful_executions / self.total_executions 
                            if self.total_executions > 0 else 0.0)
        }

class ToolExecutor:
    """Executes tool calls with proper validation and error handling"""
    
    def __init__(self, registry: ToolRegistry, default_timeout: float = 30.0):
        """
        Initialize the tool executor
        
        Args:
            registry: Registry of available tools
            default_timeout: Default timeout in seconds for tool execution
        """
        self.registry = registry
        self.default_timeout = default_timeout
        self.metrics: Dict[str, ExecutionMetrics] = {}
        self.thread_pool = ThreadPoolExecutor()
        
    async def execute_tool(self, tool_request: ToolRequest) -> ToolResponse:
        """
        Execute a tool with validation and error handling
        
        Args:
            tool_request: Tool request to execute
            
        Returns:
            ToolResponse with the result of the tool execution
        """
        tool_name = tool_request.name
        start_time = time.time()
        
        try:
            # Get tool from registry
            tool_info = self.registry.get_tool(tool_name)
            if not tool_info:
                logger.warning(f"Tool not found: {tool_name}")
                return ToolResponse(
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=tool_request.tool_call_id,
                    status=ToolResponseStatus.ERROR,
                    metadata={"error_type": "tool_not_found"}
                )
                
            # Validate arguments
            validation_errors = self.registry.validate_arguments(tool_name, tool_request.arguments)
            if validation_errors:
                logger.warning(f"Validation errors for tool {tool_name}: {validation_errors}")
                return ToolResponse(
                    content=f"Error: Invalid arguments for tool '{tool_name}': {validation_errors}",
                    tool_call_id=tool_request.tool_call_id,
                    status=ToolResponseStatus.ERROR,
                    metadata={"error_type": "validation_error", "errors": validation_errors}
                )
            
            # Get tool definition and handler
            tool_def = tool_info["definition"]
            handler = tool_info["handler"]
            
            # Execute the tool
            if tool_def.is_async:
                result = await self._execute_async_tool(
                    handler, 
                    tool_request.arguments,
                    timeout=getattr(tool_def, "timeout", self.default_timeout)
                )
            else:
                result = await self._execute_sync_tool(
                    handler,
                    tool_request.arguments,
                    timeout=getattr(tool_def, "timeout", self.default_timeout)
                )
                
            # Format the result
            content = self._format_result(result)
                
            # Create response
            response = ToolResponse(
                content=content,
                tool_call_id=tool_request.tool_call_id,
                status=ToolResponseStatus.SUCCESS,
                metadata={
                    "tool_name": tool_name,
                    "execution_time": time.time() - start_time
                }
            )
            
            # Record metrics
            self._record_metrics(tool_name, True, time.time() - start_time)
            
            return response
            
        except ToolTimeoutError as e:
            logger.error(f"Tool execution timed out: {str(e)}")
            self._record_metrics(tool_name, False, time.time() - start_time, timed_out=True)
            
            return ToolResponse(
                content=str(e),
                tool_call_id=tool_request.tool_call_id,
                status=ToolResponseStatus.TIMEOUT,
                metadata={
                    "error_type": "timeout",
                    "tool_name": tool_name,
                    "execution_time": time.time() - start_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            self._record_metrics(tool_name, False, time.time() - start_time)
            
            return ToolResponse(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                tool_call_id=tool_request.tool_call_id,
                status=ToolResponseStatus.ERROR,
                metadata={
                    "error_type": "execution_error",
                    "error_message": str(e),
                    "error_class": type(e).__name__,
                    "tool_name": tool_name,
                    "execution_time": time.time() - start_time
                }
            )
    
    async def execute_tools(self, tool_requests: List[ToolRequest]) -> List[ToolResponse]:
        """
        Execute multiple tools in parallel
        
        Args:
            tool_requests: List of tool requests to execute
            
        Returns:
            List of ToolResponse objects with results
        """
        tasks = [self.execute_tool(req) for req in tool_requests]
        return await asyncio.gather(*tasks)
    
    async def _execute_async_tool(self, handler: Callable, arguments: Dict[str, Any], timeout: float) -> Any:
        """
        Execute an async tool with timeout
        
        Args:
            handler: Async function to execute
            arguments: Arguments to pass to the function
            timeout: Timeout in seconds
            
        Returns:
            Result of the function
            
        Raises:
            ToolTimeoutError: If execution times out
        """
        try:
            return await asyncio.wait_for(handler(**arguments), timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolTimeoutError(handler.__name__, timeout)
    
    async def _execute_sync_tool(self, handler: Callable, arguments: Dict[str, Any], timeout: float) -> Any:
        """
        Execute a synchronous tool with timeout in a thread pool
        
        Args:
            handler: Synchronous function to execute
            arguments: Arguments to pass to the function
            timeout: Timeout in seconds
            
        Returns:
            Result of the function
            
        Raises:
            ToolTimeoutError: If execution times out
        """
        loop = asyncio.get_running_loop()
        
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self.thread_pool,
                    lambda: handler(**arguments)
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise ToolTimeoutError(handler.__name__, timeout)
    
    def _format_result(self, result: Any) -> str:
        """
        Format a result as a string
        
        Args:
            result: Result to format
            
        Returns:
            Formatted result as a string
        """
        if isinstance(result, str):
            return result
        elif isinstance(result, (dict, list, tuple)):
            try:
                return json.dumps(result, ensure_ascii=False, indent=2)
            except:
                return str(result)
        else:
            return str(result)
    
    def _record_metrics(self, tool_name: str, success: bool, duration: float, timed_out: bool = False):
        """
        Record metrics for a tool execution
        
        Args:
            tool_name: Name of the tool
            success: Whether the execution was successful
            duration: Execution time in seconds
            timed_out: Whether the execution timed out
        """
        if tool_name not in self.metrics:
            self.metrics[tool_name] = ExecutionMetrics()
            
        self.metrics[tool_name].record_execution(success, duration, timed_out)
    
    def get_metrics(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a specific tool or all tools
        
        Args:
            tool_name: Name of the tool to get metrics for, or None for all tools
            
        Returns:
            Dictionary of metrics
        """
        if tool_name:
            if tool_name in self.metrics:
                return self.metrics[tool_name].get_metrics()
            return {}
            
        return {
            name: metrics.get_metrics() 
            for name, metrics in self.metrics.items()
        }
