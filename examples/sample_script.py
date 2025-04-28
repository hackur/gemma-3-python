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

def main():
    # Get all arguments after the script name
    args = sys.argv[1:]
    
    # Create output dictionary
    output = {
        "received_args": args,
        "processed": True
    }
    
    # Print as JSON
    print(json.dumps(output))
    return 0

if __name__ == "__main__":
    sys.exit(main())