# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2025-05-15

### Added
- New `extract_graded_card` tool for analyzing graded Pokemon cards
- Dimension preservation system that maintains original image quality
- Modular code structure with specialized Pokemon card modules
- Coordinate scaling system for higher precision crops and annotations
- Comprehensive docstrings with usage examples in all modules

### Changed
- Refactored Pokemon card tools into separate modules:
  - `pokemon_card_utils.py`: Shared utilities
  - `pokemon_card_analyzer.py`: Card analysis
  - `pokemon_card_annotator.py`: Card annotation
  - `pokemon_card_extractor.py`: Graded card extraction
- Enhanced visualization system for better debugging
- Updated all documentation to reflect modular structure
- Improved test scripts to validate dimension preservation

### Fixed
- Fixed cropping inaccuracies by using original image dimensions
- Corrected resolution handling in all image processing functions
- Fixed JSON response parsing in test scripts

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