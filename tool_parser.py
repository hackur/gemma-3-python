"""
Tool Parser for Gemma 3 Proxy Server

This module implements a parser for extracting tool calling requests from model outputs.
It supports various formats and provides validation and error handling.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from json_utils import extract_json_objects, extract_json_from_markdown
from tool_framework import (
    ToolRequest, ToolRegistry, ToolError, 
    ToolValidationError, generate_tool_id
)

# Configure logging
logger = logging.getLogger("tool_parser")

class ToolParsingError(ToolError):
    """Raised when parsing a tool call fails"""
    def __init__(self, message: str, original_text: str = None):
        self.original_text = original_text
        super().__init__(message, None, 400)

class ToolFormat(str, Enum):
    """Supported formats for tool calls in model output"""
    OPENAI = "openai"  # OpenAI function call format
    JSON = "json"      # Plain JSON format
    MARKDOWN = "markdown"  # Markdown code block with JSON

class ToolParseResult(BaseModel):
    """Result of parsing a tool call from text"""
    success: bool = Field(True, description="Whether parsing was successful")
    tool_requests: List[ToolRequest] = Field(default_factory=list, description="Extracted tool requests")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Parsing errors")
    format_detected: Optional[ToolFormat] = Field(None, description="Format that was detected in the text")
    
class ToolCallPattern:
    """Patterns for identifying tool calls in model output"""
    # OpenAI function call format
    OPENAI_PATTERN = r'"tool_calls"\s*:\s*\[\s*{.*?"function"\s*:\s*{.*?"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*({.*?})'
    
    # JSON tool call (various formats)
    JSON_PATTERN = r'(?:\{|\[)\s*(?:"action"|"tool"|"function"|"name")\s*:\s*"([^"]+)".*?(?:"params"|"arguments"|"inputs")\s*:\s*({[^{}]*(?:\{[^{}]*\}[^{}]*)*})'
    
    # Markdown code block with JSON
    MARKDOWN_PATTERN = r'```(?:json|javascript|js)?\s*\n(.*?)\n\s*```'

class ToolParser:
    """Parser for extracting and validating tool calls from model outputs"""
    
    def __init__(self, registry: ToolRegistry):
        """
        Initialize the tool parser
        
        Args:
            registry: Registry of available tools for validation
        """
        self.registry = registry
        
    def parse_text(self, text: str, request_id: str = None) -> ToolParseResult:
        """
        Parse text to extract tool calls in various formats
        
        Args:
            text: Text from model output that might contain tool calls
            request_id: Optional request ID for correlation
            
        Returns:
            ToolParseResult with extracted tool requests and any errors
        """
        if not request_id:
            request_id = generate_tool_id()
            
        # Try different formats from most specific to most general
        formats_to_try = [
            (ToolFormat.MARKDOWN, self._extract_markdown_tool_calls),
            (ToolFormat.OPENAI, self._extract_openai_tool_calls),
            (ToolFormat.JSON, self._extract_json_tool_calls),
        ]
        
        result = ToolParseResult()
        
        for format_type, extract_fn in formats_to_try:
            try:
                tool_calls = extract_fn(text)
                if tool_calls:
                    result.format_detected = format_type
                    result.tool_requests = self._convert_to_tool_requests(tool_calls, request_id)
                    return result
            except Exception as e:
                logger.warning(f"Error extracting {format_type} tool calls: {str(e)}")
                result.errors.append({
                    "format": format_type,
                    "error": str(e),
                    "type": type(e).__name__
                })
        
        # No tool calls found
        if not result.errors:
            result.errors.append({
                "error": "no_tool_calls_found",
                "message": "No tool calls could be identified in the text"
            })
            
        result.success = False
        return result
    
    def parse_openai_format(self, tool_calls_data: List[Dict[str, Any]], request_id: str = None) -> ToolParseResult:
        """
        Parse tool calls in OpenAI function call format
        
        Args:
            tool_calls_data: List of tool call dictionaries in OpenAI format
            request_id: Optional request ID for correlation
            
        Returns:
            ToolParseResult with extracted tool requests and any errors
        """
        if not request_id:
            request_id = generate_tool_id()
            
        result = ToolParseResult(format_detected=ToolFormat.OPENAI)
        
        try:
            tool_requests = []
            for tc in tool_calls_data:
                if "function" in tc and "name" in tc["function"] and "arguments" in tc["function"]:
                    # Parse function arguments (could be string or dict)
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            # If not valid JSON, use as-is as a single argument
                            args = {"input": args}
                    
                    # Create tool request
                    tool_requests.append({
                        "name": tc["function"]["name"],
                        "arguments": args,
                        "tool_call_id": tc.get("id", generate_tool_id()),
                        "request_id": request_id
                    })
                else:
                    result.errors.append({
                        "error": "invalid_tool_call_format",
                        "message": f"Invalid tool call format: {tc}"
                    })
            
            result.tool_requests = [ToolRequest(**tr) for tr in tool_requests]
            result.success = len(result.tool_requests) > 0
            
        except Exception as e:
            logger.error(f"Error parsing OpenAI format tool calls: {str(e)}")
            result.errors.append({
                "error": "parsing_error",
                "message": str(e),
                "type": type(e).__name__
            })
            result.success = False
            
        return result
        
    def _extract_openai_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls in OpenAI function call format
        
        Args:
            text: Text that might contain tool calls
            
        Returns:
            List of extracted tool calls
        """
        # Look for JSON objects with OpenAI format
        try:
            # Try to parse the entire text as JSON first
            data = json.loads(text)
            if isinstance(data, dict) and "tool_calls" in data:
                return data["tool_calls"]
            
            # If it's a list, check each item
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "tool_calls" in item:
                        return item["tool_calls"]
        except json.JSONDecodeError:
            pass
        
        # Use regex to extract OpenAI format tool calls
        pattern = ToolCallPattern.OPENAI_PATTERN
        matches = re.finditer(pattern, text, re.DOTALL)
        
        tool_calls = []
        for match in matches:
            try:
                name = match.group(1)
                args_str = match.group(2)
                
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    # Clean up common issues in JSON
                    args_str = re.sub(r',\s*}', '}', args_str)  # Remove trailing commas
                    args_str = re.sub(r',\s*]', ']', args_str)  # Remove trailing commas in arrays
                    args = json.loads(args_str)
                
                tool_calls.append({
                    "id": generate_tool_id(),
                    "function": {
                        "name": name,
                        "arguments": args
                    }
                })
            except Exception as e:
                logger.warning(f"Error extracting OpenAI tool call: {str(e)}")
        
        return tool_calls
    
    def _extract_json_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls in generic JSON format
        
        Args:
            text: Text that might contain tool calls
            
        Returns:
            List of extracted tool calls in OpenAI format
        """
        # Try to parse the entire text as JSON first
        try:
            data = json.loads(text)
            
            # Check if it's already in OpenAI format
            if isinstance(data, dict):
                if "function" in data and "name" in data["function"]:
                    return [data]
                if "tool_calls" in data:
                    return data["tool_calls"]
                if "name" in data and ("params" in data or "arguments" in data):
                    # Convert to OpenAI format
                    args = data.get("params") or data.get("arguments") or {}
                    return [{
                        "id": generate_tool_id(),
                        "function": {
                            "name": data["name"],
                            "arguments": args
                        }
                    }]
            
            # If it's a list, check each item
            if isinstance(data, list):
                tool_calls = []
                for item in data:
                    if isinstance(item, dict):
                        if "function" in item and "name" in item["function"]:
                            tool_calls.append(item)
                        elif "name" in item and ("params" in item or "arguments" in item):
                            args = item.get("params") or item.get("arguments") or {}
                            tool_calls.append({
                                "id": generate_tool_id(),
                                "function": {
                                    "name": item["name"],
                                    "arguments": args
                                }
                            })
                if tool_calls:
                    return tool_calls
        except json.JSONDecodeError:
            pass
        
        # Extract JSON objects from the text
        objects = extract_json_objects(text)
        tool_calls = []
        
        for obj in objects:
            if "name" in obj and ("arguments" in obj or "params" in obj):
                name = obj["name"]
                args = obj.get("arguments") or obj.get("params") or {}
                
                tool_calls.append({
                    "id": generate_tool_id(),
                    "function": {
                        "name": name,
                        "arguments": args
                    }
                })
        
        # If no JSON objects found, try regex
        if not tool_calls:
            # Use regex to extract JSON-like tool calls
            pattern = ToolCallPattern.JSON_PATTERN
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    name = match.group(1)
                    args_str = match.group(2)
                    
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Clean up common issues in JSON
                        args_str = re.sub(r',\s*}', '}', args_str)  # Remove trailing commas
                        args_str = re.sub(r',\s*]', ']', args_str)  # Remove trailing commas in arrays
                        # Fix missing quotes on property names
                        args_str = re.sub(r'(\w+)\s*:', r'"\1":', args_str)
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            # Use as string if still can't parse
                            args = {"input": args_str}
                    
                    tool_calls.append({
                        "id": generate_tool_id(),
                        "function": {
                            "name": name,
                            "arguments": args
                        }
                    })
                except Exception as e:
                    logger.warning(f"Error extracting JSON tool call: {str(e)}")
        
        return tool_calls
    
    def _extract_markdown_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from markdown code blocks
        
        Args:
            text: Text that might contain tool calls in markdown code blocks
            
        Returns:
            List of extracted tool calls in OpenAI format
        """
        # Use the json_utils function to extract JSON from markdown
        json_objects = extract_json_from_markdown(text)
        
        tool_calls = []
        for obj in json_objects:
            # Check if the object is likely a tool call
            if "name" in obj and ("arguments" in obj or "params" in obj):
                name = obj["name"]
                args = obj.get("arguments") or obj.get("params") or {}
                
                tool_calls.append({
                    "id": generate_tool_id(),
                    "function": {
                        "name": name,
                        "arguments": args
                    }
                })
        
        # If no objects found with extract_json_from_markdown, fallback to original method
        if not tool_calls:
            # Extract code blocks from markdown
            code_blocks = extract_code_blocks(text)
            
            for block in code_blocks:
                try:
                    # Try to parse the block as JSON
                    try:
                        data = json.loads(block)
                    except json.JSONDecodeError:
                        # Clean up common issues in JSON
                        block = re.sub(r',\s*}', '}', block)  # Remove trailing commas
                        block = re.sub(r',\s*]', ']', block)  # Remove trailing commas in arrays
                        # Try to fix missing quotes on property names
                        block = re.sub(r'(\w+)\s*:', r'"\1":', block)
                        try:
                            data = json.loads(block)
                        except json.JSONDecodeError:
                            continue  # Skip this block if it's not valid JSON
                    
                    # Check if it's a tool call format
                    if isinstance(data, dict):
                        if "name" in data and ("arguments" in data or "params" in data):
                            name = data["name"]
                            args = data.get("arguments") or data.get("params") or {}
                            
                            tool_calls.append({
                                "id": generate_tool_id(),
                                "function": {
                                    "name": name,
                                    "arguments": args
                                }
                            })
                except Exception as e:
                    logger.warning(f"Error extracting markdown tool call: {str(e)}")
        
        return tool_calls
    
    def _convert_to_tool_requests(self, tool_calls: List[Dict[str, Any]], request_id: str) -> List[ToolRequest]:
        """
        Convert tool calls to ToolRequest objects with validation
        
        Args:
            tool_calls: List of tool calls in OpenAI format
            request_id: Request ID for correlation
            
        Returns:
            List of ToolRequest objects
        """
        tool_requests = []
        
        for tc in tool_calls:
            try:
                if "function" in tc and "name" in tc["function"] and "arguments" in tc["function"]:
                    tool_name = tc["function"]["name"]
                    args = tc["function"]["arguments"]
                    tool_call_id = tc.get("id", generate_tool_id())
                    
                    # Validate args against tool schema if tool exists
                    tool_info = self.registry.get_tool(tool_name)
                    if tool_info:
                        validation_errors = self.registry.validate_arguments(tool_name, args)
                        if validation_errors:
                            logger.warning(f"Validation errors for tool {tool_name}: {validation_errors}")
                            # Consider whether to fail or continue with warnings
                            
                    tool_requests.append(ToolRequest(
                        name=tool_name,
                        arguments=args,
                        tool_call_id=tool_call_id,
                        request_id=request_id
                    ))
            except Exception as e:
                logger.error(f"Error converting tool call to ToolRequest: {str(e)}")
        
        return tool_requests

# Utility functions for text parsing

# This function has been moved to json_utils.py and is now imported

def extract_code_blocks(text: str, language: str = None) -> List[str]:
    """
    Extract code blocks from markdown text
    
    Args:
        text: Markdown text with possible code blocks
        language: Optional language filter (e.g., 'json', 'javascript')
        
    Returns:
        List of extracted code block contents
    """
    if language:
        pattern = rf'```(?:{language})\s*\n(.*?)\n\s*```'
    else:
        pattern = r'```(?:\w+)?\s*\n(.*?)\n\s*```'
        
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(1).strip() for match in matches]

def clean_tool_arguments(args_text: str) -> Dict[str, Any]:
    """
    Clean and parse tool arguments text to valid JSON
    
    Args:
        args_text: Text containing tool arguments
        
    Returns:
        Parsed arguments as dictionary
    """
    # Common fixes for JSON generated by LLMs
    cleaned = args_text.strip()
    
    # Fix trailing/missing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix missing quotes on keys
    cleaned = re.sub(r'(\w+)(\s*:)', r'"\1"\2', cleaned)
    
    # Fix single quotes used instead of double quotes
    cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse arguments after cleaning: {str(e)}")
        raise ToolParsingError(f"Invalid JSON in arguments: {str(e)}", args_text)
