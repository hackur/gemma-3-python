# Gemma3 OpenAI-Compatible Server

An OpenAI-compatible server implementation for the Gemma 3 language model with function calling capabilities.

## Features

- OpenAI-compatible API endpoints
- Function calling support
- Python script execution
- System information retrieval
- Streaming responses
- Configurable model parameters
- Comprehensive logging

## Installation

```bash
# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Configuration

The server looks for the following files:
- `gemma-3-4b-it-q4_0/gemma-3-4b-it-q4_0.gguf` - Main model file
- `prompts/system_prompt.md` - System prompt template

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

3. Execute Python Script
```bash
curl http://127.0.0.1:1337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-it",
    "messages": [
      {"role": "user", "content": "Run the sample script with argument 'test'"}
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
            },
            "venv_path": {
              "type": "string",
              "description": "Optional path to virtual environment"
            }
          },
          "required": ["script_name"]
        }
      }
    ]
  }'
```

## Project Structure

```
.
├── gemma3.py              # Main server implementation
├── utils/
│   └── function_handler.py # Function execution utilities
├── prompts/
│   └── system_prompt.md   # System prompt template
├── scripts/               # Directory for Python scripts
├── examples/
│   └── sample_script.py   # Example script
└── tests/                # Test directory
```

## Function Calling

The server supports function calling similar to OpenAI's implementation. Functions are defined in the request and can be called by the model when appropriate.

Available functions:
1. `execute_python` - Execute Python scripts
2. `get_system_info` - Retrieve system information

## Changelog

### Version 0.1.0 (2025-04-28)

Initial release with the following features:
- OpenAI-compatible API implementation
- Function calling support
- Python script execution
- System information retrieval
- Streaming support
- Basic rate limiting
- Error handling
- Logging system
- Example implementations

## License

MIT License
```

Okay, let's break down how to create a robust script for function calling with a local gemma-3-4b-it model on your powerful M3 Max MacBook Pro, focusing on executing your execute_python function accurately and handling the results effectively.

The "perfect" script involves several key components: setting up the local environment, crafting the right prompt for Gemma, parsing its response reliably, executing the function, and handling the output.

1. Environment Setup (Local Gemma)

Choose an Inference Framework: You need a way to run gemma-3-4b-it locally. Popular choices for macOS with Apple Silicon (M3 Max) include:

Ollama: Very user-friendly. It simplifies downloading, managing, and running local models, often with Metal GPU acceleration out-of-the-box. Likely the easiest starting point.

Install Ollama: brew install ollama (if using Homebrew) or download from ollama.ai.

Pull the Gemma model: ollama pull gemma:2b-instruct (Adjust the tag if 3-4b-it becomes available or use a suitable alternative like the instruct-tuned version). Check Ollama's library for the exact available tags.

Llama.cpp: More low-level, highly performant, requires compilation but offers fine-grained control. Supports Metal. You'd typically use its Python bindings (llama-cpp-python).

Hugging Face Transformers: Requires more manual setup for the inference pipeline but offers great flexibility. Ensure you use versions that support bitsandbytes or quanto for quantization (if needed) and Metal (mps device).

Python Environment: Create a dedicated virtual environment for your project to manage dependencies. Your execute_python function definition implies it might need its own separate venv (py_script_venv), which your main script will manage.

2. The Core Script Logic (Python)

This script will orchestrate the interaction with the local Gemma model.

import json
import subprocess # To potentially call Ollama CLI or run the target script
import requests   # If using Ollama's REST API
import os         # For path management

# --- Configuration ---
# Depending on your chosen framework (Ollama API, llama-cpp-python, etc.)
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
MODEL_NAME = "gemma:2b-instruct" # Replace with your specific Gemma model tag in Ollama
PYTHON_EXECUTABLE = "/usr/bin/python3" # Or choose a specific one

# Tool Schema (as provided)
TOOLS = [
  {
    "name": "execute_python",
    "description": "Execute a pre-configured python script.",
    "parameters": {
      "type": "object",
      "properties": {
        "py_script_name": {
          "type": "string",
          "description": "The exact name of the python script file to execute (e.g., 'my_script.py')."
        },
        "py_script_args": {
          "type": "string",
          "description": "A string containing space-separated arguments for the script (e.g., '--input data.csv --output results.json')."
        },
        "py_script_venv": {
          "type": "string",
          "description": "The path to the python virtual environment activate script needed for the target script (e.g., 'path/to/venv/bin/activate'). Leave empty if none."
        }
      },
      "required": [
        "py_script_name",
        "py_script_args",
        "py_script_venv"
      ]
    }
  }
]

# --- Helper Functions ---

def format_prompt_for_gemma(user_query, tools):
    """
    Formats the prompt for Gemma, including tool descriptions.
    NOTE: The exact format might need tuning based on the specific Gemma model's
    fine-tuning for function calling. Check model documentation or experiment.
    Common patterns involve special tokens or structured text/JSON blocks.
    """
    # Example simple formatting (adjust as needed):
    prompt = f"You have access to the following tool:\n\n"
    prompt += f"Tool Name: {tools[0]['name']}\n"
    prompt += f"Description: {tools[0]['description']}\n"
    prompt += f"Parameters: {json.dumps(tools[0]['parameters'], indent=2)}\n\n"
    prompt += "Based on the user query, if the tool is appropriate, respond ONLY with a JSON object containing 'tool_name' and 'arguments' (matching the required parameters). Otherwise, respond naturally.\n\n"
    prompt += f"User Query: {user_query}"
    return prompt

def call_local_gemma(prompt, model_name):
    """Calls the local Gemma model (using Ollama API example)."""
    try:
        # Using Ollama REST API
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False, # Get the full response at once
            "format": "json", # Request JSON output IF the model supports it well for this
            "options": {
                "temperature": 0.1 # Lower temperature for more deterministic tool use
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Parse the response
        response_data = response.json()
        # The actual content is often in response_data['response'] for Ollama generate
        model_output_str = response_data.get('response', '')

        # Attempt to parse the model's output string as JSON directly
        try:
            # Find the JSON block if it's embedded
            json_start = model_output_str.find('{')
            json_end = model_output_str.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 potential_json = model_output_str[json_start:json_end]
                 parsed_output = json.loads(potential_json)
                 # Basic validation: Check if it looks like our tool call format
                 if "tool_name" in parsed_output and "arguments" in parsed_output:
                     return parsed_output # Successfully parsed tool call
                 else:
                    # It was JSON, but not the format we expected
                    return model_output_str # Return the raw string
            else:
                 # No JSON block found
                 return model_output_str # Return the raw string

        except json.JSONDecodeError:
            # Model didn't output valid JSON, return its raw response
            print("LLM did not return valid JSON.")
            return model_output_str

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def execute_local_python_script(script_name, script_args_str, venv_activate_path):
    """
    Executes the specified python script locally.
    Returns a dictionary with stdout, stderr, and return code.
    """
    # Basic security check: Ensure script_name is just a filename, not a path traversal attempt
    if "/" in script_name or "\\" in script_name or ".." in script_name:
        return {"error": "Invalid script name.", "stdout": "", "stderr": "Invalid script name.", "return_code": -1}

    # Assume scripts are in a designated 'scripts' subdirectory for safety
    script_path = os.path.abspath(os.path.join("scripts", script_name))
    scripts_dir = os.path.abspath("scripts")

    if not script_path.startswith(scripts_dir) or not os.path.isfile(script_path):
         return {"error": f"Script '{script_name}' not found or invalid.", "stdout": "", "stderr": f"Script not found: {script_path}", "return_code": -1}

    args_list = script_args_str.split() # Split args string into a list

    command = []
    if venv_activate_path and os.path.isfile(venv_activate_path):
        # Construct command to run within the venv
        # This is tricky across shells. A common way is `source venv/bin/activate && python script.py args`
        # Using subprocess might require `bash -c 'source ... && python ...'`
        # Ensure venv_activate_path is valid and safe.
        safe_venv_path = os.path.abspath(venv_activate_path)
        # Add basic check for safety if needed
        command = [
            '/bin/bash', '-c',
            f'source "{safe_venv_path}" && "{PYTHON_EXECUTABLE}" "{script_path}" {" ".join(args_list)}'
         ]
    else:
        # Run with the default python executable
        command = [PYTHON_EXECUTABLE, script_path] + args_list

    print(f"Executing command: {' '.join(command)}") # Log the command for debugging

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit code
            timeout=300 # 5-minute timeout
        )
        return {
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
            "return_code": process.returncode
        }
    except subprocess.TimeoutExpired:
         return {"error": "Script execution timed out.", "stdout": "", "stderr": "Timeout", "return_code": -1}
    except Exception as e:
        print(f"Error executing script: {e}")
        return {"error": f"Failed to execute script: {e}", "stdout": "", "stderr": str(e), "return_code": -1}

# --- Main Interaction Loop ---

user_request = "Please run the analysis script 'data_processor.py' with input 'input.csv' and use the 'prod_env/bin/activate' virtual environment." # Example user request

# 1. Format the prompt
prompt = format_prompt_for_gemma(user_request, TOOLS)
print(f"--- Sending Prompt ---\n{prompt}\n--------------------")

# 2. Call the LLM
llm_response = call_local_gemma(prompt, MODEL_NAME)
print(f"--- LLM Response ---\n{llm_response}\n------------------")

# 3. Parse and Execute Tool if requested
tool_result = None
if isinstance(llm_response, dict) and llm_response.get("tool_name") == "execute_python":
    print("--- Tool Call Detected ---")
    args = llm_response.get("arguments", {})
    # Validate required arguments (redundant if schema is enforced well, but good practice)
    if all(key in args for key in TOOLS[0]['parameters']['required']):
        try:
            tool_result = execute_local_python_script(
                script_name=args.get("py_script_name"),
                script_args_str=args.get("py_script_args", ""), # Provide default empty string
                venv_activate_path=args.get("py_script_venv")
            )
            print(f"--- Tool Execution Result ---\n{json.dumps(tool_result, indent=2)}\n--------------------------")
        except Exception as e:
            print(f"Error during tool execution logic: {e}")
            tool_result = {"error": f"Script execution failed: {e}"}
    else:
        print("Error: LLM did not provide all required arguments.")
        tool_result = {"error": "Missing required arguments from LLM."}
        # Potentially provide feedback to the LLM here in a multi-turn scenario
else:
    # LLM responded naturally or with an unexpected format
    print("--- Natural Language Response ---")
    # Output the direct LLM response if it wasn't a tool call
    print(llm_response)


# 4. (Optional) Respond back to LLM with tool result for summarization
# If you need the LLM to process the script's output (e.g., summarize results):
# if tool_result:
#     # Format the tool result clearly for the LLM
#     tool_output_prompt = f"Tool execution completed.\nResult:\n{json.dumps(tool_result)}"
#     # You might need specific tokens for tool output, e.g., <tool_outputs> ... </tool_outputs>
#
#     # Combine with previous context + tool output prompt and call LLM again
#     # final_prompt = ... (original prompt + llm response + tool output prompt) ...
#     # final_response = call_local_gemma(final_prompt, MODEL_NAME)
#     # print(f"--- Final LLM Response ---\n{final_response}\n----------------------")


3. Key Elements for "Perfection" and Accuracy

Prompt Engineering (Crucial):

Clear Instructions: Explicitly tell Gemma how to format its response when it wants to call the tool. Requesting JSON output ("format": "json" in the Ollama API call, or specific instructions in the prompt) is highly recommended for reliable parsing.

Include Schema: Always provide the tool's JSON schema within the prompt context, as shown in format_prompt_for_gemma. Add descriptions to parameters to help the model understand them.

Gemma-Specific Formatting: Research or experiment with the exact format gemma-3-4b-it expects for tool calls. Some models use specific control tokens (e.g., <tool_code>, <tool_call>) or require a specific JSON structure. The example above uses a generic JSON approach; you may need to adapt it.

Reliable Parsing:

The call_local_gemma function attempts to parse the LLM response as JSON. This is more robust than using complex regex.

Include error handling (try...except json.JSONDecodeError) in case the model doesn't return valid JSON.

Validate the parsed JSON structure (does it have tool_name and arguments?).

Robust Execution (execute_local_python_script):

Security: Implement checks to prevent execution of arbitrary scripts or path traversal (e.g., limiting scripts to a specific directory, sanitizing script_name).

Environment Handling: Carefully construct the command to correctly activate the specified virtual environment (py_script_venv) before running the target script. Using bash -c 'source ... && python ...' is a common pattern but can have nuances.

Error Capturing: Capture stdout, stderr, and the return_code from the subprocess. This provides structured output.

Timeout: Add a timeout to prevent runaway scripts.

Structured Return: The execute_local_python_script function returns a dictionary. This structured data is easy to log, potentially show to the user, or feed back to the LLM if needed.

Model Configuration: Use a low temperature (e.g., 0.0 to 0.2) when calling the LLM for function calling tasks. This makes the output more focused and deterministic, reducing the chance of creative but incorrect tool usage.

Iteration: Achieving "perfect" execution often requires testing and refinement. Log the prompts, LLM responses, and execution results to identify where things go wrong and adjust the prompt or parsing logic accordingly.

4. Running the Script

Save the orchestrator script (e.g., gemma_caller.py).

Create a subdirectory named scripts.

Place your target Python script (e.g., data_processor.py) inside the scripts directory.

Ensure Ollama (or your chosen framework) is running with the Gemma model loaded.

Run the orchestrator script: python gemma_caller.py

This detailed approach provides a solid foundation for reliable function calling with your local Gemma model on macOS. Remember to adapt the prompt formatting and LLM calling mechanism based on the specific behavior of gemma-3-4b-it and your chosen inference framework.