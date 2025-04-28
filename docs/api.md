# API Documentation

## Overview

The Gemma3 API Server provides an OpenAI-compatible API for interacting with Google's Gemma 3 language model. The server implements key OpenAI API endpoints with support for function calling.

## Base URL

```
http://127.0.0.1:1337/v1
```

## Authentication

Authentication is not required for local development. For production deployments, implement appropriate authentication mechanisms.

## Endpoints

### List Models

```http
GET /models
```

Lists available models.

**Response**

```json
{
  "data": [
    {
      "id": "gemma-3-4b-it",
      "object": "model",
      "created": 1745874646,
      "owned_by": "local"
    }
  ]
}
```

### Create Chat Completion

```http
POST /chat/completions
```

Creates a chat completion.

**Request Body**

```json
{
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
}
```

**Parameters**

- `model` (string, required): Model ID to use
- `messages` (array, required): Array of messages in the conversation
- `functions` (array, optional): Array of function definitions
- `temperature` (number, optional): Sampling temperature (0-1)
- `top_p` (number, optional): Nucleus sampling parameter (0-1)
- `n` (integer, optional): Number of completions to generate
- `stream` (boolean, optional): Whether to stream responses
- `max_tokens` (integer, optional): Maximum tokens to generate
- `presence_penalty` (number, optional): Presence penalty (-2 to 2)
- `frequency_penalty` (number, optional): Frequency penalty (-2 to 2)

**Response**

```json
{
  "id": "chatcmpl-1745874650",
  "object": "chat.completion",
  "created": 1745874650,
  "model": "gemma-3-4b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response content",
        "function_call": {
          "name": "function_name",
          "arguments": "function arguments"
        }
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 178,
    "completion_tokens": 34,
    "total_tokens": 212
  }
}
```

### Streaming Response

When `stream=true`, the response is streamed as server-sent events:

```http
data: {"id":"chatcmpl-1745874650","object":"chat.completion.chunk","created":1745874650,"model":"gemma-3-4b-it","choices":[{"index":0,"delta":{"role":"assistant","content":"Partial"},"finish_reason":null}]}

data: {"id":"chatcmpl-1745874650","object":"chat.completion.chunk","created":1745874650,"model":"gemma-3-4b-it","choices":[{"index":0,"delta":{"content":" response"},"finish_reason":null}]}

data: {"id":"chatcmpl-1745874650","object":"chat.completion.chunk","created":1745874650,"model":"gemma-3-4b-it","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Built-in Functions

### 1. execute_python

Executes a Python script with specified arguments.

**Parameters**

```json
{
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
```

### 2. get_system_info

Retrieves system information.

**Parameters**

```json
{
  "type": "object",
  "properties": {
    "info_type": {
      "type": "string",
      "enum": ["cpu", "memory", "disk", "all"]
    }
  },
  "required": ["info_type"]
}
```

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 429: Rate Limit Exceeded
- 500: Internal Server Error

Error responses follow this format:

```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "param": null,
    "code": null
  }
}
```

## Rate Limiting

Basic rate limiting is implemented:
- 100 requests per minute per IP
- Streaming requests count as one request
- Function calls count as part of the original request

## Performance

- Context window: 8192 tokens
- Metal acceleration on Apple Silicon
- Streaming latency: ~100ms
- Function calling overhead: ~50ms

## Best Practices

1. Use streaming for long responses
2. Keep prompts concise
3. Implement proper error handling
4. Monitor token usage
5. Cache frequent responses
6. Use appropriate batch sizes
7. Set reasonable timeouts