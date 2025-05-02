import os
import json
import time
import logging
import httpx
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, field_validator
import requests
from typing import List, Dict, Optional, Union, Any
import mimetypes
import re

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemma_proxy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gemma_proxy")

# Security Configuration
ALLOWED_EXTENSIONS = {'.txt', '.md', '.py', '.json', '.yaml', '.yml', '.ini', '.conf'}
MAX_FILE_SIZE = 1024 * 1024  # 1MB
ALLOWED_ROOT_DIRS = {
    str(Path.cwd()),  # Current working directory
    str(Path.home())  # User's home directory
}

# LM Studio configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
API_KEY = os.getenv("LM_STUDIO_KEY", "default-key")

# FastAPI app setup
app = FastAPI(
    title="Gemma Tool Call Proxy",
    description="Intercepts and handles problematic tool calls for Gemma 3 in LM Studio",
    version="0.1.0",
    servers=[{"url": "http://localhost:1338", "description": "Local Development"}]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

# Rate limiting configuration (requests per minute)
RATE_LIMIT = 60

class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, str]

class ChatMessage(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[ToolCall]] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096

class SafePath:
    """Security wrapper for file path operations"""
    
    @staticmethod
    def is_safe_path(path: str) -> bool:
        """Check if a path is safe to access"""
        try:
            resolved_path = str(Path(path).resolve())
            return any(
                resolved_path.startswith(allowed_dir)
                for allowed_dir in ALLOWED_ROOT_DIRS
            )
        except Exception:
            return False
            
    @staticmethod
    def is_allowed_extension(path: str) -> bool:
        """Check if file extension is allowed"""
        return Path(path).suffix.lower() in ALLOWED_EXTENSIONS
        
    @staticmethod
    def get_safe_content(path: str, max_size: int = MAX_FILE_SIZE) -> Optional[str]:
        """Safely read file content with size limit"""
        if not os.path.exists(path):
            return None
            
        file_size = os.path.getsize(path)
        if file_size > max_size:
            raise ValueError(f"File exceeds maximum size of {max_size} bytes")
            
        mime_type = mimetypes.guess_type(path)[0]
        if mime_type and not mime_type.startswith(('text/', 'application/json')):
            raise ValueError(f"Unsupported file type: {mime_type}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

class ToolArguments(BaseModel):
    """Base class for tool arguments"""
    pass

class ReadFileArguments(ToolArguments):
    path: str
    startLineNumberBaseZero: Optional[int] = None
    endLineNumberBaseZero: Optional[int] = None

    @field_validator('path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not SafePath.is_safe_path(v):
            raise ValueError("Unsafe file path")
        if not SafePath.is_allowed_extension(v):
            raise ValueError("File extension not allowed")
        return v

class ToolCallEnhanced(BaseModel):
    id: str
    name: str
    type: str = "function"
    arguments: Union[ReadFileArguments, Dict[str, Any]]

    @field_validator('arguments', mode='before')
    @classmethod
    def validate_arguments(cls, v: Union[str, Dict[str, Any]], info) -> Union[ReadFileArguments, Dict[str, Any]]:
        values = info.data
        if 'name' in values and values['name'] == 'read_file':
            if isinstance(v, str):
                v = json.loads(v)
            return ReadFileArguments(**v)
        return v

class ToolResponse(BaseModel):
    role: str = "tool"
    content: str
    tool_call_id: str
    metadata: Optional[Dict[str, Any]] = None

def format_file_content(content: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Format file content for model consumption"""
    lines = content.splitlines()
    
    if start_line is not None and end_line is not None:
        lines = lines[start_line:end_line + 1]
        
    return "\n".join(lines)

def execute_safe_tool(tool_call: ToolCallEnhanced) -> ToolResponse:
    """Execute allowed tools with enhanced validation and response formatting"""
    if tool_call.name == "read_file":
        try:
            args = tool_call.arguments
            if isinstance(args, ReadFileArguments):
                content = SafePath.get_safe_content(args.path)
                if content is None:
                    return ToolResponse(
                        content=f"Error: File {args.path} not found",
                        tool_call_id=tool_call.id,
                        metadata={"error": "file_not_found"}
                    )
                
                formatted_content = format_file_content(
                    content,
                    args.startLineNumberBaseZero,
                    args.endLineNumberBaseZero
                )
                
                return ToolResponse(
                    content=formatted_content,
                    tool_call_id=tool_call.id,
                    metadata={
                        "path": args.path,
                        "size": len(content),
                        "line_count": len(content.splitlines()),
                        "start_line": args.startLineNumberBaseZero,
                        "end_line": args.endLineNumberBaseZero
                    }
                )
        except Exception as e:
            return ToolResponse(
                content=f"Error reading file: {str(e)}",
                tool_call_id=tool_call.id,
                metadata={"error": str(e)}
            )
    
    return ToolResponse(
        content=f"Tool {tool_call.name} not allowed",
        tool_call_id=tool_call.id,
        metadata={"error": "tool_not_allowed"}
    )

async def process_tool_calls(messages: List[Dict]) -> List[Dict]:
    """Process and execute valid tool calls with enhanced validation"""
    last_message = messages[-1]
    
    if "tool_calls" in last_message:
        tool_responses = []
        for tool_call in last_message["tool_calls"]:
            try:
                validated_call = ToolCallEnhanced(**tool_call)
                response = execute_safe_tool(validated_call)
                tool_responses.append(response.dict())
            except ValidationError as e:
                logger.error(f"Invalid tool call: {str(e)}")
                tool_responses.append({
                    "role": "tool",
                    "content": f"Error: Invalid tool call - {str(e)}",
                    "tool_call_id": tool_call.get("id", ""),
                    "metadata": {"error": "validation_error"}
                })
                
        return messages + tool_responses
    
    return messages

class ConversationContext:
    """Manages conversation history and tool execution context"""
    def __init__(self, max_context_length: int = 4096):
        self.max_context_length = max_context_length
        self.tool_history: List[Dict[str, Any]] = []
        self.error_count: Dict[str, int] = {}
        
    def add_tool_result(self, tool_call: ToolCallEnhanced, result: ToolResponse):
        """Track tool execution results"""
        self.tool_history.append({
            "tool": tool_call.dict(),
            "result": result.dict(),
            "timestamp": time.time()
        })
        
    def track_error(self, tool_name: str, error: str):
        """Track tool execution errors"""
        if tool_name not in self.error_count:
            self.error_count[tool_name] = 0
        self.error_count[tool_name] += 1
        
    def should_retry(self, tool_name: str, max_retries: int = 3) -> bool:
        """Determine if a failed tool call should be retried"""
        return self.error_count.get(tool_name, 0) < max_retries
        
    def prune_context(self, messages: List[Dict]) -> List[Dict]:
        """Prune conversation context to stay within limits"""
        total_length = sum(len(str(m)) for m in messages)
        if total_length <= self.max_context_length:
            return messages
            
        # Keep system message if present
        system_message = next((m for m in messages if m["role"] == "system"), None)
        preserved = [system_message] if system_message else []
        
        # Keep recent messages up to limit
        remaining_length = self.max_context_length - len(str(preserved))
        recent_messages = []
        
        for message in reversed(messages):
            msg_length = len(str(message))
            if remaining_length - msg_length > 0:
                recent_messages.insert(0, message)
                remaining_length -= msg_length
            else:
                break
                
        return preserved + recent_messages

class ChatCompletionHandler:
    """Handles chat completion requests with retry logic"""
    def __init__(self, lm_studio_url: str, api_key: str):
        self.lm_studio_url = lm_studio_url
        self.api_key = api_key
        self.context = ConversationContext()
        
    async def handle_request(self, request: ChatRequest) -> Dict[str, Any]:
        """Process chat completion request with tool handling"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            request_data = request.dict(exclude_unset=True)
            pruned_messages = self.context.prune_context(request_data["messages"])
            request_data["messages"] = pruned_messages
            
            response = await self._make_lm_studio_request(request_data, headers)
            completion = response.json()
            
            if "choices" in completion:
                messages = pruned_messages.copy()
                messages.append(completion["choices"][0]["message"])
                
                try:
                    updated_messages = await process_tool_calls(messages)
                    
                    if len(updated_messages) > len(messages):
                        follow_up_data = {
                            **request_data,
                            "messages": self.context.prune_context(updated_messages)
                        }
                        
                        follow_up_response = await self._make_lm_studio_request(
                            follow_up_data, 
                            headers,
                            is_retry=True
                        )
                        return follow_up_response.json()
                except Exception as e:
                    logger.error(f"Tool execution error: {str(e)}")
                    # Add error context to the conversation
                    messages.append({
                        "role": "system",
                        "content": f"Error executing tool: {str(e)}"
                    })
                    
            return completion
            
        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"error": str(e), "type": type(e).__name__}
            )
            
    async def _make_lm_studio_request(
        self, 
        data: Dict[str, Any], 
        headers: Dict[str, str],
        is_retry: bool = False
    ) -> requests.Response:
        """Make request to LM Studio with retry logic"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.lm_studio_url}/chat/completions",
                    json=data,
                    headers=headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response
                
        except httpx.RequestError as e:
            if not is_retry:
                logger.warning(f"LM Studio request failed, retrying: {str(e)}")
                return await self._make_lm_studio_request(data, headers, True)
            raise HTTPException(
                status_code=502,
                detail="Error communicating with LM Studio"
            )

# Add this before the chat completion endpoint
async def _fetch_models():
    """Internal helper to fetch models with retries"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{LM_STUDIO_URL}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=30.0
            )
            if response.status_code == 404:
                # Models endpoint not supported, return default model list≠≠≠≠
                return {
                    "data": [
                        {
                            "id": "gemma-3-4b-it",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local"
                        }
                    ]
                }
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.warning(f"Error fetching models, using fallback: {str(e)}")
        return {
            "data": [
                {
                    "id": "gemma-3-4b-it",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local"
                }
            ]
        }

# Update the models endpoint to use the helper
@app.get("/v1/models")
async def list_models():
    """List available models endpoint that proxies to LM Studio"""
    return await _fetch_models()

# Add legacy models endpoint support
@app.get("/v/models")
async def legacy_list_models():
    """Legacy models endpoint that redirects to v1/models"""
    return await list_models()

# Update the FastAPI endpoint to use the new handler
handler = ChatCompletionHandler(LM_STUDIO_URL, API_KEY)

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    """Enhanced chat completion endpoint with proper error handling"""
    return await handler.handle_request(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1338) # , reload=True)
    # uvicorn.run(app, host="0.0.0.0", port=1338)
