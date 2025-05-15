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

import uuid
from datetime import datetime, timedelta
import json
from logging.handlers import RotatingFileHandler
import platform
import psutil

# Configure detailed logging with rotation and structured format
LOG_FILE = 'gemma_proxy.log'
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatter for structured logging
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _format_log(self, level, message, **kwargs):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        }
        if kwargs:
            log_data.update(kwargs)
        return json.dumps(log_data)
    
    def info(self, message, **kwargs):
        self.logger.info(self._format_log('INFO', message, **kwargs))
        
    def error(self, message, **kwargs):
        self.logger.error(self._format_log('ERROR', message, **kwargs))
        
    def warning(self, message, **kwargs):
        self.logger.warning(self._format_log('WARNING', message, **kwargs))
        
    def debug(self, message, **kwargs):
        self.logger.debug(self._format_log('DEBUG', message, **kwargs))

logger = StructuredLogger("gemma_proxy")

def generate_request_id():
    """Generate a unique request ID for correlation"""
    return str(uuid.uuid4())

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

# Error handling, monitoring and rate limiting configuration
RATE_LIMIT = 60
rate_limit_store: Dict[str, List[float]] = {}
validation_errors: Dict[str, List[Dict[str, Any]]] = {}

class ValidationMonitor:
    """Monitors and analyzes validation errors"""
    def __init__(self):
        self.errors: Dict[str, List[Dict[str, Any]]] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_cleanup = time.time()
        self.cleanup_interval = 3600  # 1 hour
        
    def track_error(self, request_id: str, error: Dict[str, Any]):
        """Track a validation error with context"""
        if request_id not in self.errors:
            self.errors[request_id] = []
        
        error_type = error.get("type", "unknown")
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.errors[request_id].append({
            **error,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Periodically cleanup old errors
        self._cleanup_old_errors()
        
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": self.error_counts,
            "recent_errors": {
                rid: errors[-5:] for rid, errors in self.errors.items()
                if errors  # Only include non-empty error lists
            }
        }
        
    def get_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common error types"""
        return sorted(
            [
                {"type": error_type, "count": count}
                for error_type, count in self.error_counts.items()
            ],
            key=lambda x: x["count"],
            reverse=True
        )[:limit]
        
    def _cleanup_old_errors(self):
        """Remove errors older than cleanup_interval"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
            
        self.last_cleanup = current_time
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        # Clean up old errors while keeping error counts accurate
        for request_id in list(self.errors.keys()):
            old_errors = [
                e for e in self.errors[request_id]
                if datetime.fromisoformat(e["timestamp"]) < cutoff_time
            ]
            for error in old_errors:
                error_type = error.get("type", "unknown")
                self.error_counts[error_type] = max(
                    0, self.error_counts.get(error_type, 0) - 1
                )
            
            # Remove error entries for this request if all are old
            current_errors = [
                e for e in self.errors[request_id]
                if datetime.fromisoformat(e["timestamp"]) >= cutoff_time
            ]
            if not current_errors:
                del self.errors[request_id]
            else:
                self.errors[request_id] = current_errors

# Initialize validation monitor
validation_monitor = ValidationMonitor()

class ValidationError422(HTTPException):
    """Custom exception for validation errors with detailed context"""
    def __init__(self, detail: str, errors: List[Dict[str, Any]], request_id: str):
        super().__init__(
            status_code=422,
            detail={
                "error": "Validation Error",
                "message": detail,
                "errors": errors,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

def track_validation_error(request_id: str, error: Dict[str, Any]):
    """Track validation errors for analysis"""
    if request_id not in validation_errors:
        validation_errors[request_id] = []
    validation_errors[request_id].append({
        **error,
        "timestamp": datetime.utcnow().isoformat()
    })

def check_rate_limit(identifier: str) -> bool:
    """Check if request should be rate limited"""
    current_time = time.time()
    # Clean up old entries
    rate_limit_store[identifier] = [
        t for t in rate_limit_store.get(identifier, [])
        if current_time - t < 60  # Remove entries older than 1 minute
    ]
    # Check current rate
    if len(rate_limit_store[identifier]) >= RATE_LIMIT:
        return False
    # Add new request timestamp
    rate_limit_store[identifier].append(current_time)
    return True

# Security headers and rate limiting middleware
@app.middleware("http")
async def middleware(request: Request, call_next):
    request_id = generate_request_id()
    start_time = time.time()

    try:
        # Add request ID to state
        request.state.request_id = request_id
        
        # Add security headers and correlation ID
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-Request-ID"] = request_id
        
        # Rate limiting
        client_ip = request.client.host
        if not check_rate_limit(client_ip):
            logger.warning(
                "Rate limit exceeded",
                request_id=request_id,
                client_ip=client_ip,
                rate_limit=RATE_LIMIT
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": f"Rate limit of {RATE_LIMIT} requests per minute exceeded",
                    "request_id": request_id
                }
            )
        
        # Log request completion
        duration = time.time() - start_time
        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            duration=duration,
            status_code=response.status_code
        )
        
        return response
        
    except Exception as e:
        # Log unhandled errors
        duration = time.time() - start_time
        logger.error(
            "Unhandled error in middleware",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            duration=duration
        )
        
        # Convert to HTTP exception if needed
        if not isinstance(e, HTTPException):
            e = HTTPException(
                status_code=500,
                detail={
                    "error": "Internal Server Error",
                    "message": str(e),
                    "request_id": request_id
                }
            )
        
        return JSONResponse(
            status_code=e.status_code,
            content=e.detail
        )

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
        request_id = generate_request_id()
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            request_data = request.dict(exclude_unset=True)
            
            # Log request details
            logger.info(
                "Processing chat request",
                request_id=request_id,
                model=request_data.get("model"),
                message_count=len(request_data["messages"])
            )
            
            # Context pruning
            original_length = len(request_data["messages"])
            pruned_messages = self.context.prune_context(request_data["messages"])
            request_data["messages"] = pruned_messages
            
            if len(pruned_messages) < original_length:
                logger.info(
                    "Context pruned",
                    request_id=request_id,
                    original_messages=original_length,
                    pruned_messages=len(pruned_messages)
                )
            
            # Make request to LM Studio
            response = await self._make_lm_studio_request(request_data, headers)
            completion = response.json()
            
            if "choices" in completion:
                messages = pruned_messages.copy()
                messages.append(completion["choices"][0]["message"])
                
                try:
                    # Log tool call processing
                    tool_calls = completion["choices"][0]["message"].get("tool_calls", [])
                    if tool_calls:
                        logger.info(
                            "Processing tool calls",
                            request_id=request_id,
                            tool_count=len(tool_calls)
                        )
                    
                    # Process tool calls
                    updated_messages = await process_tool_calls(messages)
                    
                    if len(updated_messages) > len(messages):
                        # Tool calls were executed
                        logger.info(
                            "Tool calls executed",
                            request_id=request_id,
                            tools_executed=len(updated_messages) - len(messages)
                        )
                        
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
                    logger.error(
                        "Tool execution failed",
                        request_id=request_id,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    # Add error context to the conversation
                    messages.append({
                        "role": "system",
                        "content": f"Error executing tool: {str(e)}"
                    })
            
            # Log completion
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                request_id=request_id,
                duration=duration,
                has_choices="choices" in completion
            )
            
            return completion
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                duration=duration
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "type": type(e).__name__,
                    "request_id": request_id
                }
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

# Health and monitoring endpoints
@app.get("/v1/health")
async def health_check():
    """Get service health status and monitoring information"""
    memory = psutil.Process().memory_info()
    cpu_percent = psutil.Process().cpu_percent()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "uptime": time.time() - psutil.Process().create_time(),
        "system": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "memory_usage_mb": memory.rss / (1024 * 1024),
            "cpu_percent": cpu_percent
        },
        "monitoring": {
            "validation_errors": validation_monitor.get_error_stats(),
            "rate_limits": {
                "current_connections": len(rate_limit_store),
                "limit": RATE_LIMIT
            }
        }
    }

# Monitoring endpoints
@app.get("/v1/monitoring/validation-errors")
async def get_validation_errors():
    """Get validation error statistics and analysis"""
    stats = validation_monitor.get_error_stats()
    common_errors = validation_monitor.get_common_errors()
    
    return {
        "statistics": stats,
        "common_errors": common_errors,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/v1/monitoring/error-types")
async def get_error_types():
    """Get breakdown of error types and frequencies"""
    return {
        "error_types": validation_monitor.error_counts,
        "total_errors": sum(validation_monitor.error_counts.values()),
        "timestamp": datetime.utcnow().isoformat()
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
async def chat_completion(request: Request):
    """Enhanced chat completion endpoint with proper error handling"""
    request_id = request.state.request_id
    start_time = time.time()
    
    try:
        # Parse and validate request body
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            error = {
                "location": "body",
                "type": "json_decode_error",
                "detail": str(e)
            }
            track_validation_error(request_id, error)
            raise ValidationError422(
                detail="Invalid JSON in request body",
                errors=[error],
                request_id=request_id
            )

        # Log raw request
        logger.info(
            "Received chat completion request",
            request_id=request_id,
            headers=dict(request.headers),
            body=body
        )
        
        # Validate request schema
        try:
            validated_request = ChatRequest(**body)
        except ValidationError as e:
            # Track each validation error
            errors = []
            for error in e.errors():
                error_info = {
                    "location": " -> ".join(str(loc) for loc in error["loc"]),
                    "type": error["type"],
                    "detail": error["msg"]
                }
                errors.append(error_info)
                track_validation_error(request_id, error_info)
                
            raise ValidationError422(
                detail="Request validation failed",
                errors=errors,
                request_id=request_id
            )
        
        # Process request
        response = await handler.handle_request(validated_request)
        
        # Log success
        duration = time.time() - start_time
        logger.info(
            "Chat completion success",
            request_id=request_id,
            duration=duration
        )
        
        return response
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (including our ValidationError422)
        raise
        
    except Exception as e:
        # Handle unexpected errors
        duration = time.time() - start_time
        logger.error(
            "Chat completion error",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
            duration=duration
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": str(e),
                "type": type(e).__name__,
                "request_id": request_id
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1338) # , reload=True)
    # uvicorn.run(app, host="0.0.0.0", port=1338)
