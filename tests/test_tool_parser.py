"""
Unit tests for the tool parser module.

This module contains tests for parsing tool calls in different formats from model outputs.
"""

import os
import sys
import json
import pytest

# Add parent directory to path to import tool modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool_framework import ToolRegistry, ToolRequest, ToolCategory
from tool_parser import ToolParser, ToolFormat, ToolParseResult

# Sample tool definitions
SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        },
        "category": "utility"
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        },
        "category": "web"
    }
]

# Dummy handlers for tools
def get_weather_handler(location, units="celsius"):
    """Dummy handler for get_weather tool"""
    return f"Weather in {location} is 22 degrees {units}"

def search_web_handler(query, num_results=5):
    """Dummy handler for search_web tool"""
    return [f"Result {i} for {query}" for i in range(min(num_results, 10))]

# Sample model outputs for testing
OPENAI_FORMAT_OUTPUT = """
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo-0613",
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 31,
    "total_tokens": 87
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "I'll help you find the weather in San Francisco.",
        "tool_calls": [
          {
            "id": "call_abc123",
            "function": {
              "name": "get_weather",
              "arguments": {
                "location": "San Francisco",
                "units": "celsius"
              }
            },
            "type": "function"
          }
        ]
      },
      "finish_reason": "tool_calls",
      "index": 0
    }
  ]
}
"""

MARKDOWN_FORMAT_OUTPUT = """
To answer your question about the weather, I'll need to check the current conditions.

```json
{
  "name": "get_weather",
  "arguments": {
    "location": "San Francisco",
    "units": "celsius"
  }
}
```

This will give us the current weather data for San Francisco.
"""

JSON_FORMAT_OUTPUT = """
I'll search for information about Python programming.

{"tool": "search_web", "params": {"query": "Python programming tutorial", "num_results": 3}}

This should return some helpful tutorials on Python programming.
"""

MULTIPLE_TOOLS_OUTPUT = """
I'll help you with both of your requests:

First, let's check the weather:
```json
{
  "name": "get_weather",
  "arguments": {
    "location": "New York",
    "units": "fahrenheit"
  }
}
```

And then search for some information:
{"tool": "search_web", "params": {"query": "best places to visit in New York"}}
"""

MALFORMED_JSON_OUTPUT = """
I'll help you with the weather.

{"name": "get_weather", "arguments": {"location": "Paris", units: "celsius"}}

Let me know if you need anything else!
"""

class TestToolParser:
    """Tests for the ToolParser class"""
    
    @pytest.fixture
    def registry(self):
        """Create a registry with sample tools for testing"""
        registry = ToolRegistry()
        
        for tool in SAMPLE_TOOLS:
            handler = get_weather_handler if tool["name"] == "get_weather" else search_web_handler
            registry.register_tool(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["parameters"],
                handler_fn=handler,
                category=tool["category"],
                required=tool["parameters"].get("required", [])
            )
            
        return registry
        
    @pytest.fixture
    def parser(self, registry):
        """Create a tool parser for testing"""
        return ToolParser(registry)
    
    def test_parse_openai_format(self, parser):
        """Test parsing tool calls in OpenAI format"""
        result = parser.parse_text(OPENAI_FORMAT_OUTPUT)
        
        assert result.success
        assert result.format_detected == ToolFormat.OPENAI
        assert len(result.tool_requests) == 1
        assert result.tool_requests[0].name == "get_weather"
        assert result.tool_requests[0].arguments == {"location": "San Francisco", "units": "celsius"}
    
    def test_parse_markdown_format(self, parser):
        """Test parsing tool calls in markdown code blocks"""
        result = parser.parse_text(MARKDOWN_FORMAT_OUTPUT)
        
        assert result.success
        assert result.format_detected == ToolFormat.MARKDOWN
        assert len(result.tool_requests) == 1
        assert result.tool_requests[0].name == "get_weather"
        assert result.tool_requests[0].arguments == {"location": "San Francisco", "units": "celsius"}
    
    def test_parse_json_format(self, parser):
        """Test parsing tool calls in simple JSON format"""
        result = parser.parse_text(JSON_FORMAT_OUTPUT)
        
        assert result.success
        assert result.format_detected == ToolFormat.JSON
        assert len(result.tool_requests) == 1
        assert result.tool_requests[0].name == "search_web"
        assert result.tool_requests[0].arguments["query"] == "Python programming tutorial"
        assert result.tool_requests[0].arguments["num_results"] == 3
    
    def test_parse_multiple_tools(self, parser):
        """Test parsing multiple tool calls in different formats"""
        result = parser.parse_text(MULTIPLE_TOOLS_OUTPUT)
        
        assert result.success
        assert len(result.tool_requests) > 0
        # The exact parsing behavior for multiple tools depends on implementation
        # We should find at least one tool
    
    def test_parse_malformed_json(self, parser):
        """Test handling malformed JSON in tool calls"""
        result = parser.parse_text(MALFORMED_JSON_OUTPUT)
        
        # We expect the parser to try to recover from common errors
        assert result.success
        assert len(result.tool_requests) == 1
        assert result.tool_requests[0].name == "get_weather"
        assert result.tool_requests[0].arguments["location"] == "Paris"
    
    def test_extract_json_objects(self, parser):
        """Test extracting JSON objects from text"""
        # This test depends on how the extract_json_objects utility is implemented
        from tool_parser import extract_json_objects
        
        text = """
        Here are some results:
        {"name": "foo", "value": 42}
        Also: {"items": [1, 2, 3]}
        """
        
        objects = extract_json_objects(text)
        assert len(objects) == 2
        assert objects[0]["name"] == "foo"
        assert objects[1]["items"] == [1, 2, 3]
    
    def test_extract_code_blocks(self, parser):
        """Test extracting code blocks from markdown text"""
        from tool_parser import extract_code_blocks
        
        text = """
        Here's some Python code:
        ```python
        def hello():
            print("Hello world")
        ```
        
        And some JSON:
        ```json
        {"name": "example"}
        ```
        """
        
        all_blocks = extract_code_blocks(text)
        assert len(all_blocks) == 2
        
        json_blocks = extract_code_blocks(text, language="json")
        assert len(json_blocks) == 1
        assert "example" in json_blocks[0]
    
    def test_clean_tool_arguments(self, parser):
        """Test cleaning and parsing tool arguments text"""
        from tool_parser import clean_tool_arguments
        
        # Test fixing various common issues in LLM-generated JSON
        args_text = '{location: "New York", "units": "fahrenheit",}'
        args = clean_tool_arguments(args_text)
        
        assert args["location"] == "New York"
        assert args["units"] == "fahrenheit"

if __name__ == "__main__":
    pytest.main()
