# Gemma3 OpenAI-Compatible Server

An OpenAI-compatible API server implementation for Google's Gemma 3 language model with function calling capabilities, optimized for Apple Silicon.

## Features

- OpenAI-compatible API endpoints
- Function calling support with robust validation
- Python script execution with sandbox support
- System information retrieval with real-time monitoring
- Streaming responses for long-running operations
- Metal GPU acceleration for Apple Silicon
- Comprehensive logging and error tracking
- Memory-efficient conversation handling
- Built-in benchmarking suite
- Extensive test coverage

## Installation

```bash
# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Download model (if not already present)
mkdir -p gemma-3-4b-it-q4_0
# Download gemma-3-4b-it-q4_0.gguf into the above directory
```

## Project Structure

```
.
├── gemma3.py              # Main server implementation
├── benchmarks/            # Performance benchmarking tools
│   ├── __init__.py
│   └── benchmark_runner.py
├── utils/
│   ├── __init__.py
│   ├── function_handler.py # Function execution utilities
│   └── monitoring.py      # System monitoring utilities
├── prompts/
│   └── system_prompt.md   # System prompt template
├── scripts/              # Directory for Python scripts
│   └── sample_script.py   # Example script
├── tests/               # Test directory
│   ├── __init__.py
│   ├── test_api.py      # API endpoint tests
│   ├── test_functions.py # Function calling tests
│   └── test_model.py    # Model behavior tests
├── examples/            # Example implementations
│   ├── sample_script.py
│   └── test.sh
├── docs/               # Documentation
│   ├── api.md         # API documentation
│   ├── functions.md   # Function calling guide
│   └── development.md # Development guide
└── requirements.txt    # Project dependencies
```

## Configuration

The server can be configured through environment variables or a config file:

- `GEMMA_MODEL_PATH`: Path to model file (default: gemma-3-4b-it-q4_0/gemma-3-4b-it-q4_0.gguf)
- `GEMMA_SYSTEM_PROMPT`: Path to system prompt (default: prompts/system_prompt.md)
- `GEMMA_CONTEXT_SIZE`: Model context size (default: 8192)
- `GEMMA_NUM_THREADS`: Number of inference threads (default: 4)
- `GEMMA_API_HOST`: API host address (default: 127.0.0.1)
- `GEMMA_API_PORT`: API port (default: 1337)

## Usage

Start the server:

```bash
uv run python gemma3.py
```

The server will be available at `http://127.0.0.1:1337/v1`

### API Endpoints

1. List Models
```bash
curl http://127.0.0.1:1337/v1/models
```

2. Chat Completion
```bash
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
```

## Function Calling

The server supports an OpenAI-compatible function calling implementation with these built-in functions:

1. `execute_python` - Execute Python scripts in a controlled environment
   - Parameters:
     - `script_name`: Name of the script to execute
     - `arguments`: Command line arguments (optional)
     - `venv_path`: Virtual environment path (optional)

2. `get_system_info` - Retrieve system information
   - Parameters:
     - `info_type`: Type of information to retrieve ("cpu", "memory", "disk", or "all")

### Adding Custom Functions

Custom functions can be added by implementing a function handler:

```python
from utils.function_handler import register_function

@register_function
def custom_function(param1: str, param2: int) -> dict:
    """
    Function documentation
    """
    # Implementation
    return {"result": "value"}
```

## Benchmarking

Run the benchmark suite:

```bash
python -m benchmarks.benchmark_runner
```

This will test:
- Model loading time
- Inference latency
- Memory usage
- Function calling overhead
- Response streaming performance

## Testing

Run the test suite:

```bash
pytest tests/
```

## Development

See [Development Guide](docs/development.md) for:
- Code style guide
- Pull request process
- Testing guidelines
- Documentation requirements

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Run tests and benchmarks
5. Submit a pull request

See [Contributing Guide](CONTRIBUTING.md) for details.