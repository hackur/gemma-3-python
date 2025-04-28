#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Cleaning up any existing processes on port 1337${NC}"
lsof -ti:1337 | xargs kill -9 2>/dev/null || echo -e "${RED}No existing process found on port 1337${NC}"

# Wait a moment for process cleanup
sleep 1

echo -e "${BLUE}Testing Gemma3 API Server${NC}"

# Test 1: List Models
echo -e "\n${GREEN}Test 1: Listing Available Models${NC}"
curl http://127.0.0.1:1337/v1/models

# Test 2: Basic Chat Completion
echo -e "\n\n${GREEN}Test 2: Basic Chat Completion${NC}"
curl http://127.0.0.1:1337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-it",
    "messages": [
      {"role": "user", "content": "What is the current CPU usage?"}
    ],
    "functions": [
      {
        "name": "get_system_info",
        "description": "Get system information",
        "parameters": {
          "type": "object",
          "properties": {
            "info_type": {
              "type": "string",
              "enum": ["cpu", "memory", "disk", "all"]
            }
          },
          "required": ["info_type"]
        }
      }
    ]
  }'

# Test 3: Execute Python Script
echo -e "\n\n${GREEN}Test 3: Execute Python Script${NC}"
curl http://127.0.0.1:1337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-it",
    "messages": [
      {"role": "user", "content": "Run the sample script with argument test"}
    ],
    "functions": [
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
            }
          },
          "required": ["script_name"]
        }
      }
    ]
  }'

# Test 4: Get All System Information
echo -e "\n\n${GREEN}Test 4: Get All System Information${NC}"
curl http://127.0.0.1:1337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-it",
    "messages": [
      {"role": "user", "content": "Get all system information including CPU, memory, and disk usage"}
    ],
    "functions": [
      {
        "name": "get_system_info",
        "description": "Get system information",
        "parameters": {
          "type": "object",
          "properties": {
            "info_type": {
              "type": "string",
              "enum": ["cpu", "memory", "disk", "all"]
            }
          },
          "required": ["info_type"]
        }
      }
    ]
  }'