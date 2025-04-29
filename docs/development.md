# Development Guide

This guide provides information for developers contributing to the Gemma3 API Server project.

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return types
- Document all functions and classes using Google-style docstrings
- Keep line length to 88 characters (Black formatter standard)
- Use meaningful variable and function names

## Project Organization

### Core Components

1. API Server (`gemma3.py`)
   - FastAPI application setup
   - Route definitions
   - Request/response models
   - Error handling

2. Function Handler (`utils/function_handler.py`)
   - Function registration system
   - Execution environment management
   - Error handling and logging

3. Monitoring (`utils/monitoring.py`)
   - System resource monitoring
   - Performance metrics collection
   - Health checks

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest fixtures for common test setups

To run the tests, you need Python 3.9 or higher and the following dependencies:

To manage the virtual environment and install dependencies, it is recommended to use `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pytest pytest-asyncio pytest-cov
```

Example commands for running tests:

```bash
# Run test suite
pytest tests/

# Run benchmarks
python -m benchmarks.benchmark_runner

# Run specific test file
pytest tests/test_api.py
```

To run the tests, you need Python 3.9 or higher and the following dependencies:

```bash
pip install pytest pytest-asyncio pytest-cov
```

Example commands for running tests:

```bash
# Run test suite
pytest tests/

# Run benchmarks
python -m benchmarks.benchmark_runner

# Run specific test file
pytest tests/test_api.py
```
- Mock external dependencies appropriately
- Test both success and error cases

### Documentation

- Keep README.md up to date
- Document all API endpoints
- Provide examples for new features
- Update CHANGELOG.md for all changes

## Development Workflow

1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

2. Implement Changes
- Write tests first (TDD approach)
- Implement functionality
- Add documentation
- Run linter and formatter

3. Test Changes
```bash
# Run test suite
pytest tests/

# Run benchmarks
python -m benchmarks.benchmark_runner

# Run specific test file
pytest tests/test_api.py
```

4. Submit Pull Request
- Update CHANGELOG.md
- Ensure all tests pass
- Add documentation
- Request review

## Performance Considerations

1. Model Optimization
- Use Metal acceleration on Apple Silicon
- Optimize batch sizes for inference
- Monitor memory usage

2. API Performance
- Use async/await for I/O operations
- Implement proper caching
- Monitor response times

3. Function Calling
- Implement timeouts for all operations
- Use resource limits for script execution
- Monitor system resource usage

## Security Guidelines

1. Input Validation
- Validate all API inputs
- Sanitize file paths
- Validate script arguments

2. Script Execution
- Run scripts in isolated environment
- Implement timeouts
- Limit system resource usage

3. Error Handling
- Never expose internal errors
- Log errors appropriately
- Return user-friendly messages

## Monitoring and Logging

1. Application Logs
- Use structured logging
- Include request IDs
- Log appropriate detail level

2. Metrics
- Track response times
- Monitor memory usage
- Record error rates

3. Alerts
- Set up alerts for error spikes
- Monitor system resource usage
- Track model performance

## Adding New Features

1. Function Implementation
```python
from utils.function_handler import register_function

@register_function
def new_function(param1: str, param2: int) -> dict:
    """
    Implementation of new functionality.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        dict: Result of the operation
    """
    # Implementation
    return {"result": "value"}
```

2. Test Implementation
```python
def test_new_function():
    """Test new functionality"""
    result = new_function("test", 123)
    assert result["status"] == "success"
    assert "result" in result
```

3. Documentation
- Add function description to README.md
- Update API documentation
- Add example usage

## Common Issues

1. Model Loading
- Ensure model file exists
- Check file permissions
- Verify GPU acceleration

2. Script Execution
- Check script permissions
- Verify Python environment
- Monitor timeouts

3. Memory Usage
- Monitor model memory usage
- Check for memory leaks
- Implement proper cleanup

## Release Process

1. Version Update
- Update version in pyproject.toml
- Update CHANGELOG.md
- Tag release in git

2. Testing
- Run full test suite
- Run benchmarks
- Test in staging environment

3. Documentation
- Update API documentation
- Update version numbers
- Review release notes

## Support

For questions or issues:
1. Check existing issues
2. Create detailed bug report
3. Include relevant logs
4. Provide reproduction steps