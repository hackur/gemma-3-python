# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-05-15

### Added
- New `annotate_pokemon_card` tool for identifying and labeling card parts
- Enhanced image analysis with Pokemon card detection
- Support for multiple annotation styles (colors and box types)
- Automated test scripts for Pokemon card analysis and annotation
- Smart cropping functionality with focus area detection
- Timestamped output directories for test results
- Comprehensive documentation for image processing tools

### Changed
- Updated README with image processing tools documentation
- Improved error handling in image processing functions
- Enhanced project structure with new test files

### Fixed
- Fixed syntax errors in smart_crop_image function
- Corrected tool registration in gemma_proxy.py

## [0.1.0] - 2025-04-28

### Added
- Initial release with OpenAI-compatible API
- Function calling support with Python script execution
- System information retrieval functionality
- Streaming API responses
- Basic rate limiting implementation
- Comprehensive logging system
- Example scripts and tests
- Support for Gemma 3 4B-IT quantized model

### Changed
- Optimized model loading for Apple Silicon
- Improved error handling and validation

### Fixed
- Memory management for long conversations
- Thread safety in conversation logging