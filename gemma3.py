import os
import asyncio
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        # Create new conversation or get existing
        conv_id = conversation_logger.create_conversation()
        logger.info(f"Starting conversation {conv_id}")
        
        # Prepare messages
        messages = request.messages
        
        # Format prompt with system message and conversation
        formatted_prompt = f"{SYSTEM_PROMPT}\n\n"
        
        # Add conversation context
        for msg in messages:
            if msg.role == "system":
                formatted_prompt += f"System: {msg.content}\n"
            elif msg.role == "user":
                formatted_prompt += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                formatted_prompt += f"Assistant: {msg.content}\n"
        
        # Add function definitions if provided
        if request.functions:
            formatted_prompt += "\nAvailable functions:\n"
            for func in request.functions:
                formatted_prompt += f"Function: {func.name}\n"
                formatted_prompt += f"Description: {func.description}\n"
                formatted_prompt += f"Parameters: {json.dumps(func.parameters, indent=2)}\n\n"
            formatted_prompt += "Remember to respond with a valid JSON function call if needed.\n"
        
        formatted_prompt += "\nAssistant:"

        # Log the formatted prompt
        logger.debug(f"Formatted prompt for conversation {conv_id}:\n{formatted_prompt}")
        
        # Generate completion with lower temperature for more focused responses
        completion = model.create_completion(
            prompt=formatted_prompt,
            temperature=0.1,  # Lower temperature for more deterministic responses
            max_tokens=2048,
            stop=["User:", "System:", "\n\n"],
        )

        response_text = completion["choices"][0]["text"].strip()
        logger.debug(f"Raw model response for conversation {conv_id}:\n{response_text}")
        
        # Parse function calls if present
        function_call = None
        if request.functions and "{" in response_text and "}" in response_text:
            try:
                # Find the JSON object in the response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                potential_json = response_text[start_idx:end_idx]
                
                # Parse and validate the JSON
                function_json = json.loads(potential_json)
                
                if "name" in function_json and "arguments" in function_json:
                    # Validate function exists
                    if function_json["name"] in [f.name for f in request.functions]:
                        # Execute function
                        function_call = {
                            "name": function_json["name"],
                            "arguments": json.dumps(function_json["arguments"])
                        }
                        logger.info(f"Executing function {function_json['name']} for conversation {conv_id}")
                        
                        result = await function_executor.execute_function(
                            function_json["name"],
                            function_json["arguments"]
                        )
                        response_text = json.dumps(result, indent=2)
                        logger.debug(f"Function result for conversation {conv_id}:\n{response_text}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse function call JSON in conversation {conv_id}: {e}")

        # Prepare response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model="gemma-3-4b-it",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                    "function_call": function_call
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(formatted_prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(formatted_prompt.split()) + len(response_text.split())
            }
        )

        # Log the interaction
        conversation_logger.log_interaction(
            conv_id,
            request.model_dump(),
            response.model_dump(),
            formatted_prompt
        )
        
        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Gemma3 API Server...")
    uvicorn.run(app, host="127.0.0.1", port=1337)