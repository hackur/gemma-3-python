"""
Unit tests for the tool framework.

This module contains tests for the tool registration system.
"""

import os
import sys
import pytest
import asyncio
from unittest.mock import MagicMock

# Add parent directory to path to import tool_framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tool_framework import (
    ToolRegistry, ToolDefinition, ToolCategory, 
    ToolRequest, ToolResponse, ToolResponseStatus,
    ToolError, ToolNotFoundError, ToolValidationError
)

# Test tool definitions
SAMPLE_READ_FILE_TOOL = {
    "name": "read_file",
    "description": "Reads content from a file",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file"
            },
            "start_line": {
                "type": "integer",
                "description": "Line number to start reading from (0-indexed)"
            },
            "end_line": {
                "type": "integer",
                "description": "Line number to end reading at (0-indexed)"
            }
        },
        "required": ["path"]
    },
    "category": "file"
}

SAMPLE_IMAGE_TOOL = {
    "name": "analyze_image",
    "description": "Analyzes an image to detect objects or text",
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "URL of the image to analyze"
            },
            "analysis_type": {
                "type": "string", 
                "enum": ["objects", "text", "faces", "labels"],
                "description": "Type of analysis to perform"
            }
        },
        "required": ["image_url"]
    },
    "category": "image"
}

def dummy_read_file_handler(path, start_line=None, end_line=None):
    """Dummy handler for read_file tool"""
    return f"Content of {path} from {start_line} to {end_line}"

async def dummy_analyze_image_handler(image_url, analysis_type="objects"):
    """Dummy async handler for analyze_image tool"""
    return {
        "results": [f"Result for {image_url} using {analysis_type}"],
        "count": 1
    }

class TestToolDefinition:
    """Tests for the ToolDefinition class"""
    
    def test_valid_tool_definition(self):
        """Test creating a valid tool definition"""
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}}
        )
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.category == ToolCategory.GENERAL
        
    def test_invalid_parameters(self):
        """Test validation of invalid parameters"""
        with pytest.raises(ValueError):
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters="not a dict"
            )
            
        with pytest.raises(ValueError):
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={}  # Missing type
            )
            
        with pytest.raises(ValueError):
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object"}  # Missing properties for object type
            )
            
class TestToolRegistry:
    """Tests for the ToolRegistry class"""
    
    def test_register_tool(self):
        """Test registering a tool"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"]
        )
        
        assert SAMPLE_READ_FILE_TOOL["name"] in registry.tools
        assert registry.tools[SAMPLE_READ_FILE_TOOL["name"]]["handler"] == dummy_read_file_handler
        assert registry.tools[SAMPLE_READ_FILE_TOOL["name"]]["definition"].name == SAMPLE_READ_FILE_TOOL["name"]
        
    def test_register_async_tool(self):
        """Test registering an async tool"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_IMAGE_TOOL["name"],
            description=SAMPLE_IMAGE_TOOL["description"],
            parameters=SAMPLE_IMAGE_TOOL["parameters"],
            handler_fn=dummy_analyze_image_handler,
            category=SAMPLE_IMAGE_TOOL["category"]
        )
        
        assert SAMPLE_IMAGE_TOOL["name"] in registry.tools
        assert registry.tools[SAMPLE_IMAGE_TOOL["name"]]["definition"].is_async == True
        
    def test_unregister_tool(self):
        """Test unregistering a tool"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"]
        )
        
        assert registry.unregister_tool(SAMPLE_READ_FILE_TOOL["name"]) == True
        assert SAMPLE_READ_FILE_TOOL["name"] not in registry.tools
        assert SAMPLE_READ_FILE_TOOL["name"] not in registry.categories.get("file", [])
        
        # Try unregistering non-existent tool
        assert registry.unregister_tool("non_existent_tool") == False
        
    def test_get_tool(self):
        """Test getting a tool by name"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"]
        )
        
        tool = registry.get_tool(SAMPLE_READ_FILE_TOOL["name"])
        assert tool is not None
        assert tool["definition"].name == SAMPLE_READ_FILE_TOOL["name"]
        assert tool["handler"] == dummy_read_file_handler
        
        # Test getting non-existent tool
        assert registry.get_tool("non_existent_tool") is None
        
    def test_get_tools_by_category(self):
        """Test getting tools by category"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"]
        )
        registry.register_tool(
            name=SAMPLE_IMAGE_TOOL["name"],
            description=SAMPLE_IMAGE_TOOL["description"],
            parameters=SAMPLE_IMAGE_TOOL["parameters"],
            handler_fn=dummy_analyze_image_handler,
            category=SAMPLE_IMAGE_TOOL["category"]
        )
        
        file_tools = registry.get_tools_by_category("file")
        assert len(file_tools) == 1
        assert file_tools[0].name == SAMPLE_READ_FILE_TOOL["name"]
        
        image_tools = registry.get_tools_by_category(ToolCategory.IMAGE)
        assert len(image_tools) == 1
        assert image_tools[0].name == SAMPLE_IMAGE_TOOL["name"]
        
        # Test empty category
        empty_tools = registry.get_tools_by_category("non_existent_category")
        assert len(empty_tools) == 0
        
    def test_get_all_tools(self):
        """Test getting all tools"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"]
        )
        registry.register_tool(
            name=SAMPLE_IMAGE_TOOL["name"],
            description=SAMPLE_IMAGE_TOOL["description"],
            parameters=SAMPLE_IMAGE_TOOL["parameters"],
            handler_fn=dummy_analyze_image_handler,
            category=SAMPLE_IMAGE_TOOL["category"]
        )
        
        all_tools = registry.get_all_tools()
        assert len(all_tools) == 2
        tool_names = [tool.name for tool in all_tools]
        assert SAMPLE_READ_FILE_TOOL["name"] in tool_names
        assert SAMPLE_IMAGE_TOOL["name"] in tool_names
        
    def test_get_openai_schema(self):
        """Test getting tools in OpenAI schema format"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"],
            required=["path"]
        )
        
        schema = registry.get_openai_schema()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == SAMPLE_READ_FILE_TOOL["name"]
        assert schema[0]["function"]["description"] == SAMPLE_READ_FILE_TOOL["description"]
        assert schema[0]["function"]["parameters"] == SAMPLE_READ_FILE_TOOL["parameters"]
        assert "required" in schema[0]["function"]
        assert schema[0]["function"]["required"] == ["path"]
        
    def test_validate_arguments(self):
        """Test validating tool arguments"""
        registry = ToolRegistry()
        registry.register_tool(
            name=SAMPLE_READ_FILE_TOOL["name"],
            description=SAMPLE_READ_FILE_TOOL["description"],
            parameters=SAMPLE_READ_FILE_TOOL["parameters"],
            handler_fn=dummy_read_file_handler,
            category=SAMPLE_READ_FILE_TOOL["category"],
            required=["path"]
        )
        
        # Valid arguments
        errors = registry.validate_arguments(
            SAMPLE_READ_FILE_TOOL["name"],
            {"path": "/path/to/file", "start_line": 0}
        )
        assert len(errors) == 0
        
        # Missing required parameter
        errors = registry.validate_arguments(
            SAMPLE_READ_FILE_TOOL["name"],
            {"start_line": 0}
        )
        assert len(errors) > 0
        assert "missing_required_parameter" in errors[0]["error"]
        
        # Non-existent tool
        errors = registry.validate_arguments(
            "non_existent_tool",
            {"path": "/path/to/file"}
        )
        assert len(errors) > 0
        assert "tool_not_found" in errors[0]["error"]
        
        # Invalid argument type
        errors = registry.validate_arguments(
            SAMPLE_READ_FILE_TOOL["name"],
            {"path": "/path/to/file", "start_line": "not an integer"}
        )
        assert len(errors) > 0
        assert "schema_validation_error" in errors[0]["error"]

if __name__ == "__main__":
    pytest.main()
