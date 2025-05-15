"""
JSON utilities for the Gemma 3 Tool Calling Framework.

This module provides utilities for working with JSON data, especially for
extracting JSON objects from text that may contain other content.
"""

import re
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger("json_utils")

def extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Extract valid JSON objects from text
    
    Args:
        text: Text that might contain JSON objects
        
    Returns:
        List of parsed JSON objects
    """
    # List to store extracted JSON objects
    objects = []
    
    # Try to find JSON objects in the text
    # This uses a regex pattern to find anything that looks like a JSON object
    pattern = r'\{(?:[^{}]|(?R))*\}'
    matches = re.finditer(r'\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}])*\})*\})*\}', text)
    
    for match in matches:
        try:
            # Try to parse the match as JSON
            json_str = match.group(0)
            
            # Clean up common issues in JSON
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            # Attempt to fix unquoted keys (a common issue in LLM outputs)
            json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)
            
            # Try to parse the JSON
            obj = json.loads(json_str)
            
            # Only add if it's a dictionary
            if isinstance(obj, dict):
                objects.append(obj)
                
        except json.JSONDecodeError:
            # If it's not valid JSON, try a more aggressive approach
            try:
                # Try to extract values with Python's eval (be careful with this in production!)
                # This approach is less secure but can handle more malformed JSON
                # For safety, you might want to disable this in production
                
                # Replace common issues
                json_str = match.group(0)
                json_str = json_str.replace("'", "\"")  # Replace single quotes with double quotes
                json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)  # Quote unquoted keys
                
                # Try again
                obj = json.loads(json_str)
                
                if isinstance(obj, dict):
                    objects.append(obj)
            except:
                # If still can't parse, log and continue
                logger.debug(f"Failed to parse JSON object: {match.group(0)[:100]}...")
    
    return objects

def extract_json_from_markdown(text: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from markdown code blocks
    
    Args:
        text: Markdown text that might contain JSON code blocks
        
    Returns:
        List of parsed JSON objects
    """
    # Try to find JSON code blocks in markdown
    pattern = r'```(?:json)?\s*\n([\s\S]*?)\n```'
    matches = re.finditer(pattern, text)
    
    objects = []
    for match in matches:
        try:
            # Try to parse the match as JSON
            json_str = match.group(1)
            
            # Clean up common issues in JSON
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
            
            obj = json.loads(json_str)
            
            # Can be a dict or a list
            if isinstance(obj, dict):
                objects.append(obj)
            elif isinstance(obj, list):
                # If it's a list, add each dict in the list
                for item in obj:
                    if isinstance(item, dict):
                        objects.append(item)
        except json.JSONDecodeError:
            logger.debug(f"Failed to parse JSON from markdown block: {match.group(1)[:100]}...")
    
    return objects

def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by fixing common issues
    
    Args:
        json_str: JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Quote unquoted keys
    json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)
    
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", "\"")
    
    return json_str
