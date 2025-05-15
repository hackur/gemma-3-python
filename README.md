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
- Advanced image analysis tools for Pokemon cards
- Smart image cropping with focus area detection
- Automated card part annotation with bounding boxes and labels

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
├── gemma_proxy.py         # Tool calling framework proxy server
├── example_tools.py       # Implementation of image processing tools
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
│   ├── test_model.py    # Model behavior tests
│   ├── test_tool_executor.py # Tool execution tests
│   ├── test_tool_framework.py # Tool framework tests
│   └── test_tool_parser.py # Tool parsing tests
├── examples/            # Example implementations
│   ├── sample_script.py
│   └── test.sh
├── docs/               # Documentation
│   ├── api.md         # API documentation
│   ├── functions.md   # Function calling guide
│   └── development.md # Development guide
├── test_analyze_pokemon.py # Test script for Pokemon card analysis
├── test_annotate_pokemon.py # Test script for Pokemon card annotation
├── test_pokemon_smart_crop.py # Test script for smart cropping Pokemon cards
├── output/             # Directory for test output (timestamped subdirectories)
│   └── ANNOTATED_CARDS--* # Timestamped directories with annotation results
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

## Image Processing Tools

The server includes several advanced image processing tools specifically designed for Pokemon card analysis:

### 1. analyze_image

Analyzes an image and returns detailed information about its contents, with special handling for Pokemon cards.

```bash
curl http://localhost:1338/v1/tools/analyze_image \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "analyze_objects": true,
    "analyze_text": false
  }'
```

### 2. smart_crop_image

Intelligently crops an image to specified dimensions while focusing on the most important area.

```bash
curl http://localhost:1338/v1/tools/smart_crop_image \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "target_width": 300,
    "target_height": 400,
    "focus_area": "center"
  }'
```

### 3. annotate_pokemon_card

Identifies and annotates parts of a Pokemon card with bounding boxes and labels.

```bash
curl http://localhost:1338/v1/tools/annotate_pokemon_card \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/pokemon_card.jpg",
    "label_color": "red",
    "box_type": "rectangle"
  }'
```

Supported options:
- **label_color**: red, green, blue, yellow, white, black
- **box_type**: rectangle, circle

## Pokemon Card Analysis Examples

The following examples demonstrate how to use the Pokemon card analysis tools in practice. The examples use the test scripts included in the repository.

### Complete Card Analysis Workflow

This example shows a complete workflow for analyzing Pokemon cards:

1. **Run the test script**:

```bash
source .venv/bin/activate
python test_annotate_pokemon.py
```

2. **Results Overview**:

The script processes both front and back Pokemon card images and generates annotated versions with different styles. Results are saved to a timestamped directory (e.g., `output/ANNOTATED_CARDS--2025-05-15-11-51-37/`).

### Example Front Card Analysis

#### Original Card (Front)

```
docs/pokenmon-card-front.webp
```

#### Annotated Card with Rectangle Boxes

<img src="output/ANNOTATED_CARDS--2025-05-15-11-51-37/pokenmon-card-front_annotated_red_rectangle.png" alt="Front Card with Red Rectangle Annotations" width="400"/>

```python
# Python code example for front card annotation
async def annotate_front_card():
    image_path = "docs/pokenmon-card-front.webp"
    image_data_uri = image_to_data_uri(image_path)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1338/v1/tools/annotate_pokemon_card",
            json={
                "image_url": image_data_uri,
                "label_color": "red",
                "box_type": "rectangle"
            }
        )
        
        # The response contains the annotated image as a data URI
        annotated_image_uri = response.text.strip('"')
        
        # Save or display the annotated image
        output_path = "output/annotated_front_card.png"
        save_data_uri_to_file(annotated_image_uri, output_path)
```

#### Front Card with Circle Annotations

<img src="output/ANNOTATED_CARDS--2025-05-15-11-51-37/pokenmon-card-front_annotated_blue_circle.png" alt="Front Card with Blue Circle Annotations" width="400"/>

### Example Back Card Analysis

#### Original Card (Back)

```
docs/pokenmon-card-back.webp.webp
```

#### Annotated Card with Rectangle Boxes

<img src="output/ANNOTATED_CARDS--2025-05-15-11-51-37/pokenmon-card-back.webp_annotated_green_rectangle.png" alt="Back Card with Green Rectangle Annotations" width="400"/>

```python
# Python code example for back card annotation
async def annotate_back_card():
    image_path = "docs/pokenmon-card-back.webp.webp"
    image_data_uri = image_to_data_uri(image_path)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1338/v1/tools/annotate_pokemon_card",
            json={
                "image_url": image_data_uri,
                "label_color": "green",
                "box_type": "rectangle"
            }
        )
        
        # Process response...
```

### Smart Cropping Example

```python
# Python code example for smart cropping
async def smart_crop_pokemon_card():
    image_path = "docs/pokenmon-card-front.webp"
    image_data_uri = image_to_data_uri(image_path)
    
    # Crop to focus on just the Pokemon image in the center
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1338/v1/tools/smart_crop_image",
            json={
                "image_url": image_data_uri,
                "target_width": 300,
                "target_height": 300,
                "focus_area": "center"
            }
        )
        
        # The response contains the cropped image as a data URI
        cropped_image_uri = response.text.strip('"')
        
        # Save or display the cropped image
        output_path = "output/cropped_pokemon.png"
        save_data_uri_to_file(cropped_image_uri, output_path)
```

### Integration Example

This example shows how to combine analysis, cropping, and annotation in a single workflow:

```python
async def complete_pokemon_card_analysis(image_path):
    """Perform complete analysis of a Pokemon card"""
    image_data_uri = image_to_data_uri(image_path)
    results = {}
    
    # Step 1: Analyze the image
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1338/v1/tools/analyze_image",
            json={
                "image_url": image_data_uri,
                "analyze_objects": True
            }
        )
        results["analysis"] = response.json()
    
    # Step 2: Smart crop the image
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1338/v1/tools/smart_crop_image",
            json={
                "image_url": image_data_uri,
                "target_width": 400,
                "target_height": 400,
                "focus_area": "center"
            }
        )
        cropped_image_uri = response.text.strip('"')
        results["cropped_image"] = cropped_image_uri
    
    # Step 3: Annotate the card
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1338/v1/tools/annotate_pokemon_card",
            json={
                "image_url": image_data_uri,
                "label_color": "red",
                "box_type": "rectangle"
            }
        )
        annotated_image_uri = response.text.strip('"')
        results["annotated_image"] = annotated_image_uri
    
    return results
```

### Output Organization

The test scripts organize outputs into timestamped directories for easy reference:

```
output/
├── ANNOTATED_CARDS--2025-05-15-11-51-37/
│   ├── pokenmon-card-front_annotated_red_rectangle.png
│   ├── pokenmon-card-front_annotated_green_rectangle.png
│   ├── pokenmon-card-front_annotated_blue_circle.png
│   ├── pokenmon-card-front_annotated_yellow_circle.png
│   ├── pokenmon-card-back.webp_annotated_red_rectangle.png
│   ├── pokenmon-card-back.webp_annotated_green_rectangle.png
│   ├── pokenmon-card-back.webp_annotated_blue_circle.png
│   ├── pokenmon-card-back.webp_annotated_yellow_circle.png
│   ├── summary.json
│   └── test_script.py
```

Each run also creates a `summary.json` file with metadata about all processed images.

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