import os
import asyncio
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama
from loguru import logger
import json
import time
from datetime import datetime
from pathlib import Path
from utils.function_handler import FunctionExecutor

# Configure logging
LOGS_DIR = Path("logs/conversations")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
logger.add(
    LOGS_DIR / "gemma3_{time}.log",
    rotation="1 day",
    retention="1 month",
    level="DEBUG"
)

# Initialize FastAPI app
app = FastAPI(title="Gemma3 API Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables and configurations
MODEL_PATH = "gemma-3-4b-it-q4_0/gemma-3-4b-it-q4_0.gguf"
SYSTEM_PROMPT_PATH = "prompts/system_prompt.md"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_THREADS = 4

class ConversationLogger:
    def __init__(self, base_dir: Path = LOGS_DIR):
        self.base_dir = base_dir
        
    def create_conversation(self) -> str:
        """Create a new conversation directory and return its ID"""
        conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_dir = self.base_dir / conv_id
        conv_dir.mkdir(exist_ok=True)
        return conv_id
        
    def log_interaction(
        self,
        conv_id: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        prompt: str = None
    ):
        """Log an interaction in the conversation"""
        conv_dir = self.base_dir / conv_id
        
        # Log request
        with open(conv_dir / "request.json", "a") as f:
            json.dump(request, f, indent=2)
            f.write("\n---\n")
            
        # Log response
        with open(conv_dir / "response.json", "a") as f:
            json.dump(response, f, indent=2)
            f.write("\n---\n")
            
        # Log prompt if provided
        if prompt:
            with open(conv_dir / "prompts.txt", "a") as f:
                f.write(f"=== {datetime.now().isoformat()} ===\n")
                f.write(prompt)
                f.write("\n---\n")

# Initialize conversation logger
conversation_logger = ConversationLogger()

# Initialize function executor
function_executor = FunctionExecutor()

# Pydantic models for API
class FunctionParameter(BaseModel):
    type: str
    description: str
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[FunctionDefinition]] = None
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

# Load system prompt
def load_system_prompt():
    if os.path.exists(SYSTEM_PROMPT_PATH):
        with open(SYSTEM_PROMPT_PATH, "r") as f:
            return f.read().strip()
    return "You are a helpful AI assistant."

SYSTEM_PROMPT = load_system_prompt()

# Initialize model with improved logging
logger.info(f"Loading model from {MODEL_PATH}")
try:
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=DEFAULT_CONTEXT_SIZE,
        n_threads=DEFAULT_THREADS,
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "gemma-3-4b-it",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Validate messages array
        if not request.messages:
            raise HTTPException(status_code=422, detail="Messages array cannot be empty")
            
        # Create new conversation or get existing
        conv_id = conversation_logger.create_conversation()
        logger.info(f"Starting conversation {conv_id}")
        
        # Format conversation into a single string
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                messages.append(f"System: {msg.content}")
            elif msg.role == "user":
                messages.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                messages.append(f"Assistant: {msg.content}")
        
        # Join messages with clear separators
        formatted_prompt = "\n".join(messages) + "\nAssistant:"

        # Handle streaming responses
        if request.stream:
            async def generate():
                completion = model.create_completion(
                    prompt=formatted_prompt,
                    temperature=request.temperature or 0.7,
                    max_tokens=2048,
                    stop=["Human:", "System:", "\n\n"],
                    stream=True
                )
                
                for token in completion:
                    chunk = {
                        "id": f"chatcmpl-{conv_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token["choices"][0]["text"]},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Send final chunk
                yield f"data: {json.dumps({'choices':[{'finish_reason':'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")

        # Generate regular completion
        completion = model.create_completion(
            prompt=formatted_prompt,
            temperature=request.temperature or 0.7,
            max_tokens=2048,
            stop=["Human:", "System:", "\n\n"],
            stream=False
        )

        response = {
            "id": f"chatcmpl-{conv_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"].strip()
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(formatted_prompt.split()),
                "completion_tokens": len(completion["choices"][0]["text"].split()),
                "total_tokens": len(formatted_prompt.split()) + len(completion["choices"][0]["text"].split())
            }
        }

        # Log the interaction
        conversation_logger.log_interaction(
            conv_id=conv_id,
            request=request.model_dump(),  # Using model_dump() instead of deprecated dict()
            response=response,
            prompt=formatted_prompt
        )

        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Gemma3 API Server...")
    uvicorn.run(app, host="127.0.0.1", port=1337)