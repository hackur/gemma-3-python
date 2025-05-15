"""
Sample script demonstrating function calling in Gemma3.

This script demonstrates how to create a function that can be called by the
Gemma3 API server. It processes command line arguments and returns results
in a JSON format that can be parsed by the server.

Examples:
    Basic usage:
        python sample_script.py --name "Test" --count 5
        
    With JSON input:
        python sample_script.py --json '{"name": "Test", "count": 5}'

Args:
    --name (str): Name parameter for demonstration
    --count (int): Count parameter for demonstration
    --json (str): Alternative JSON input format
"""

import sys
import json
import argparse
from typing import Dict, Any
from pathlib import Path

def process_data(name: str, count: int) -> Dict[str, Any]:
    """
    Process input parameters and generate a response.
    
    Args:
        name: Input name to process
        count: Number of times to repeat operation
        
    Returns:
        Dict containing processed results
        
    Example:
        >>> process_data("test", 3)
        {
            'input': {'name': 'test', 'count': 3},
            'result': ['test-1', 'test-2', 'test-3'],
            'success': True
        }
    """
    try:
        result = [f"{name}-{i+1}" for i in range(count)]
        return {
            "input": {
                "name": name,
                "count": count
            },
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main() -> int:
    """
    Main entry point for the sample script.
    
    Parses command line arguments and executes the sample processing.
    
    Returns:
        0 on success, 1 on error
        
    Example:
        python sample_script.py --name "Test" --count 5
    """
    parser = argparse.ArgumentParser(description="Sample script for Gemma3 function calling")
    parser.add_argument("--name", type=str, help="Name parameter")
    parser.add_argument("--count", type=int, help="Count parameter")
    parser.add_argument("--json", type=str, help="JSON input format")
    
    args = parser.parse_args()
    
    try:
        # Handle JSON input format
        if args.json:
            input_data = json.loads(args.json)
            name = input_data.get("name")
            count = input_data.get("count")
        else:
            name = args.name
            count = args.count
            
        # Validate input
        if not name or not count:
            raise ValueError("Both name and count parameters are required")
            
        # Process data
        result = process_data(name, count)
        
        # Output result
        print(json.dumps(result, indent=2))
        return 0
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())