"""
Tool Framework for Gemma 3 Proxy Server

This module implements a comprehensive tool calling framework that can be integrated
with the Gemma 3 proxy server. It provides classes for defining, registering,
validating, and executing tools that can be called by the Gemma 3 model.
"""

import os
import json
import time
import asyncio
import inspect
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, TypeVar, Generic, Type
from pydantic import BaseModel, Field, ValidationError, field_validator
import jsonschema

# Configure logging
logger = logging.getLogger("tool_framework")

def generate_tool_id():
    """Generate a unique ID for a tool instance"""
    import uuid
    return str(uuid.uuid4())

class ToolCategory(str, Enum):
    """Categories for organizing tools"""
    GENERAL = "general"
    FILE = "file"
    IMAGE = "image"
    WEB = "web"
    MEMORY = "memory"
    UTILITY = "utility"
    CUSTOM = "custom"

class ToolDefinition(BaseModel):
    """Schema for defining a tool that can be called by the model"""
    name: str = Field(..., description="Name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for tool parameters")
    version: str = Field("1.0.0", description="Version of the tool")
    required: List[str] = Field(default_factory=list, description="List of required parameters")
    category: ToolCategory = Field(ToolCategory.GENERAL, description="Category for organizing tools")
    is_async: bool = Field(False, description="Whether the tool is async")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for tool response")
    
    @field_validator('parameters')
    def validate_parameters_schema(cls, v):
        """Validate that parameters follow JSON Schema format"""
        # Basic validation that it's a valid JSON schema
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a JSON Schema object")
        
        # Check for required fields in JSON Schema
        if "type" not in v:
            raise ValueError("Parameters schema must include 'type' field")
        
        # If it's an object type, it should have properties
        if v.get("type") == "object" and "properties" not in v:
            raise ValueError("Object type must have 'properties' field")
            
        return v

class ToolRequest(BaseModel):
    """Schema for a tool call request from the model"""
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool call")
    tool_call_id: str = Field(..., description="ID of the tool call")
    request_id: str = Field(default_factory=generate_tool_id, description="ID of the request")

class ToolResponseStatus(str, Enum):
    """Status of a tool execution response"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    
class ToolResponse(BaseModel):
    """Schema for a tool execution response"""
    role: str = Field("tool", description="Role identifier for the response")
    content: str = Field(..., description="Content of the tool response")
    tool_call_id: str = Field(..., description="ID of the tool call this response is for")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the response")
    status: ToolResponseStatus = Field(ToolResponseStatus.SUCCESS, description="Status of the tool execution")

class ToolError(Exception):
    """Base exception for tool-related errors"""
    def __init__(self, message: str, tool_name: str = None, status_code: int = 400):
        self.tool_name = tool_name
        self.status_code = status_code
        super().__init__(message)
        
class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found"""
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found", tool_name, 404)
        
class ToolValidationError(ToolError):
    """Raised when tool arguments fail validation"""
    def __init__(self, tool_name: str, errors: List[Dict[str, Any]]):
        self.errors = errors
        message = f"Validation failed for tool '{tool_name}': {errors}"
        super().__init__(message, tool_name, 400)
        
class ToolExecutionError(ToolError):
    """Raised when tool execution fails"""
    def __init__(self, tool_name: str, original_error: Exception):
        self.original_error = original_error
        message = f"Error executing tool '{tool_name}': {str(original_error)}"
        super().__init__(message, tool_name, 500)

class ToolTimeoutError(ToolError):
    """Raised when tool execution times out"""
    def __init__(self, tool_name: str, timeout_seconds: int):
        message = f"Execution of tool '{tool_name}' timed out after {timeout_seconds} seconds"
        super().__init__(message, tool_name, 408)

class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        """Initialize an empty tool registry"""
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.categories: Dict[str, List[str]] = {}
        
    def register_tool(self, 
                     name: str, 
                     description: str, 
                     parameters: Dict[str, Any], 
                     handler_fn: Callable,
                     category: Union[str, ToolCategory] = ToolCategory.GENERAL,
                     required: List[str] = None,
                     version: str = "1.0.0",
                     **kwargs) -> None:
        """
        Register a new tool with the registry
        
        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: JSON Schema for the tool parameters
            handler_fn: Function that implements the tool
            category: Category for organizing tools
            required: List of required parameters
            version: Version of the tool
            **kwargs: Additional tool properties
        """
        # Convert string category to enum if needed
        category_value = category
        if isinstance(category, str):
            try:
                category_value = ToolCategory(category)
            except ValueError:
                category_value = ToolCategory.CUSTOM
        
        # Set default required list if not provided
        if required is None:
            required = []
            
        # Determine if the handler is async if not explicitly provided in kwargs
        if 'is_async' not in kwargs:
            kwargs['is_async'] = asyncio.iscoroutinefunction(handler_fn)
        
        try:
            # Create and validate the tool definition
            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters,
                category=category_value,
                required=required,
                version=version,
                **kwargs
            )
            
            # Store in registry with the handler function
            self.tools[name] = {
                "definition": tool_def,
                "handler": handler_fn
            }
            
            # Add to category index
            category_name = str(category_value.value if hasattr(category_value, 'value') else category_value)
            if category_name not in self.categories:
                self.categories[category_name] = []
            self.categories[category_name].append(name)
            
            logger.info(f"Registered tool: {name} (category: {category_name})")
        except ValidationError as e:
            logger.error(f"Failed to register tool '{name}': {e}")
            raise
        
    def unregister_tool(self, name: str) -> bool:
        """
        Remove a tool from the registry
        
        Args:
            name: Name of the tool to remove
            
        Returns:
            bool: True if the tool was removed, False if it wasn't found
        """
        if name in self.tools:
            # Get the tool definition and extract category
            tool_def = self.tools[name]["definition"]
            category_val = tool_def.category
            
            # Convert category to string for lookup in categories dict
            category_name = str(category_val.value if hasattr(category_val, 'value') else category_val)
            
            # Remove from category list
            if category_name in self.categories and name in self.categories[category_name]:
                self.categories[category_name].remove(name)
                
            # Remove the tool itself
            del self.tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
            
        logger.warning(f"Attempted to unregister non-existent tool: {name}")
        return False
        
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Tool information or None if not found
        """
        return self.tools.get(name)
        
    def get_tools_by_category(self, category: Union[str, ToolCategory]) -> List[ToolDefinition]:
        """
        Get all tools in a specific category
        
        Args:
            category: Category to filter by
            
        Returns:
            List[ToolDefinition]: List of tool definitions in the category
        """
        # Convert category to string for lookup in categories dict
        if isinstance(category, ToolCategory):
            category_name = str(category.value)
        else:
            category_name = str(category)
            
        tool_names = self.categories.get(category_name, [])
        return [self.tools[name]["definition"] for name in tool_names if name in self.tools]
        
    def get_all_tools(self) -> List[ToolDefinition]:
        """
        Get all registered tools
        
        Returns:
            List[ToolDefinition]: List of all tool definitions
        """
        return [tool["definition"] for tool in self.tools.values()]
        
    def get_openai_schema(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions in OpenAI function calling format
        
        Returns:
            List[Dict[str, Any]]: List of tools in OpenAI schema format
        """
        schemas = []
        for tool in self.tools.values():
            definition = tool["definition"]
            schema = {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                }
            }
            
            # Add required field if present
            if definition.required:
                schema["function"]["required"] = definition.required
                
            schemas.append(schema)
        return schemas
    
    def validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Validate arguments against a tool's parameter schema
        
        Args:
            tool_name: Name of the tool
            arguments: Arguments to validate
            
        Returns:
            List[Dict[str, Any]]: List of validation errors (empty if valid)
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return [{"error": "tool_not_found", "message": f"Tool '{tool_name}' not found"}]
            
        schema = tool["definition"].parameters
        required = tool["definition"].required
        
        errors = []
        
        # Check required parameters
        for param in required:
            if param not in arguments:
                errors.append({
                    "error": "missing_required_parameter",
                    "parameter": param,
                    "message": f"Missing required parameter: {param}"
                })
                
        # Validate against JSON schema
        try:
            jsonschema.validate(instance=arguments, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            errors.append({
                "error": "schema_validation_error",
                "message": str(e),
                "path": list(e.path) if e.path else None
            })
            
        return errors

class ToolSecurityManager:
    """Security manager for tool execution"""
    
    def __init__(self, default_rate_limit: int = 10):
        """
        Initialize the security manager
        
        Args:
            default_rate_limit: Default rate limit per minute per tool
        """
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.permissions: Dict[str, List[str]] = {}
        self.tool_limits: Dict[str, int] = {}
        self.default_rate_limit = default_rate_limit
        
    def set_tool_rate_limit(self, tool_name: str, limit: int) -> None:
        """
        Set rate limit for a specific tool
        
        Args:
            tool_name: Name of the tool
            limit: Maximum calls per minute
        """
        self.tool_limits[tool_name] = limit
        
    def check_rate_limit(self, tool_name: str, user_id: str = "anonymous") -> bool:
        """
        Check if a tool call should be rate limited
        
        Args:
            tool_name: Name of the tool being called
            user_id: ID of the user making the call
            
        Returns:
            bool: True if the call is allowed, False if it should be limited
        """
        key = f"{user_id}:{tool_name}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                "count": 0,
                "window_start": now
            }
            
        window_data = self.rate_limits[key]
        
        # Reset window if needed (60 second window)
        if now - window_data["window_start"] > 60:
            window_data["count"] = 0
            window_data["window_start"] = now
            
        # Get limit for tool (or use default)
        tool_limit = self.tool_limits.get(tool_name, self.default_rate_limit)
        
        # Check limit
        if window_data["count"] >= tool_limit:
            logger.warning(f"Rate limit reached for {key}")
            return False
            
        # Increment counter
        window_data["count"] += 1
        return True
        
    def has_permission(self, tool_name: str, user_id: str = "anonymous") -> bool:
        """
        Check if user has permission to use a tool
        
        Args:
            tool_name: Name of the tool
            user_id: ID of the user
            
        Returns:
            bool: True if the user has permission, False otherwise
        """
        # If no permissions are defined, allow all
        if not self.permissions:
            return True
            
        # Check user permissions
        if user_id not in self.permissions:
            return False
        return tool_name in self.permissions[user_id] or "*" in self.permissions[user_id]
        
    def grant_permission(self, tool_name: str, user_id: str) -> None:
        """
        Grant permission to use a tool
        
        Args:
            tool_name: Name of the tool
            user_id: ID of the user
        """
        if user_id not in self.permissions:
            self.permissions[user_id] = []
        if tool_name not in self.permissions[user_id]:
            self.permissions[user_id].append(tool_name)
            
    def revoke_permission(self, tool_name: str, user_id: str) -> None:
        """
        Revoke permission to use a tool
        
        Args:
            tool_name: Name of the tool
            user_id: ID of the user
        """
        if user_id in self.permissions and tool_name in self.permissions[user_id]:
            self.permissions[user_id].remove(tool_name)
            
    def grant_all_permissions(self, user_id: str) -> None:
        """
        Grant permission to use all tools
        
        Args:
            user_id: ID of the user
        """
        self.permissions[user_id] = ["*"]
        
    def clear_permissions(self, user_id: str) -> None:
        """
        Clear all permissions for a user
        
        Args:
            user_id: ID of the user
        """
        if user_id in self.permissions:
            del self.permissions[user_id]
"""
Tool Executor class will be implemented in the next phase (Task 5).
This file provides the core tool registration system.
"""
