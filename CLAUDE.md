# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`crossword-grid-extraction` is a Python-based computer vision tool that extracts crossword grid structures from images. It uses OpenCV to detect, straighten, and digitize crossword puzzles into binary matrices (0=black cells, 1=white cells). The tool automatically detects grid dimensions and can export results to CSV.

## Development Commands

### Installation
```bash
# Install dependencies
uv pip install -e .

# Install with dev dependencies (pytest, mypy, ruff)
uv pip install -e ".[dev]"
```

### Running the Application
The application uses `argparse` for CLI and can be invoked directly:

```bash
# Extract and straighten the grid
python src/crossword.py --input crosswords1.jpg extract --output grid.jpg

# Detect grid dimensions (columns x rows)
python src/crossword.py --input crosswords1.jpg size

# Full conversion: extract, detect, convert to CSV (with dot detection)
python src/crossword.py --input crosswords1.jpg convert --output grid.csv

# Disable dot detection (output only 0=black, 1=white)
python src/crossword.py --input crosswords1.jpg convert --output grid.csv --no-detect-dots

# Use custom intensity threshold
python src/crossword.py --input crosswords1.jpg convert --output grid.csv --threshold 150

# Enable visualization mode (shows detected contours)
python src/crossword.py --input crosswords1.jpg extract --visualize

# Enable verbose logging (TRACE level)
python src/crossword.py --input crosswords1.jpg --verbose convert --output grid.csv

# Get help
python src/crossword.py --help
python src/crossword.py --input image.jpg convert --help
```

### Development Tools
```bash
# Type checking
mypy src/crossword.py

# Linting
ruff check src/crossword.py

# Run tests (when available)
pytest
```

### Environment Setup
- Python version: 3.11+
- Uses `uv` for dependency management
- Dependencies declared in `pyproject.toml`

### Key Dependencies
All dependencies are declared in `pyproject.toml`:
- `opencv-python>=4.8.0` - Image processing and computer vision
- `numpy>=1.24.0` - Numerical operations and matrix manipulation
- `scipy>=1.10.0` - Signal processing (peak detection for grid lines)
- `loguru>=0.7.0` - Structured logging

Note: The CLI uses Python's built-in `argparse` module (no external dependency required).

## Architecture

### Module Structure

The codebase is organized into two main modules:

1. **`src/extract.py`** - Core image processing library
   - Contains all computer vision algorithms and grid extraction logic
   - Designed to be reusable by multiple interfaces (CLI, MCP server, etc.)
   - Exports: `extract_grid()`, `detect_grid_dimensions()`, `convert_to_matrix()`, `save_matrix_to_csv()`
   - Exceptions: `GridExtractionError`, `DimensionDetectionError`

2. **`src/crossword.py`** - CLI interface
   - Provides command-line interface using argparse
   - Imports and calls functions directly from `extract.py`
   - Handles argument parsing, logging configuration, and error handling
   - Command handlers: `cmd_extract()`, `cmd_size()`, `cmd_convert()`
   - Helper functions: `load_image()`, `_create_visualization()`
   - Main entry point: `main()` function

### Processing Pipeline

The application follows a three-stage pipeline:

1. **Grid Extraction** (`extract_grid` in `extract.py`)
   - Applies adaptive thresholding to handle shadows/lighting variations
   - Detects the largest quadrilateral contour (the crossword boundary)
   - Performs perspective transform to straighten the grid
   - Returns warped image with computed dimensions

2. **Dimension Detection** (`detect_grid_dimensions` in `extract.py`)
   - Uses projection profiles (sum of pixel intensities along axes)
   - Applies peak detection on inverted projections to find grid lines
   - Calculates median cell size from detected peaks
   - Includes bimodal distribution refinement to handle double-detected grid lines
   - Cross-validation: uses successfully detected dimension to validate the other
   - Estimates rows/columns by dividing image dimensions by median cell size
   - Robust to missing or faint grid lines through median-based approach

3. **Grid Conversion** (`convert_to_matrix` in `extract.py`)
   - Slices straightened image into individual cells based on detected dimensions
   - Samples center region of each cell (configurable margin) to avoid grid line interference
   - Classifies cells based on intensity threshold:
     - 0 = black cell (filled/blocked)
     - 1 = white cell (empty)
     - 2 = white cell with black dot in bottom-right corner (solution letter marker)
   - Dot detection examines bottom-right corner region for dark pixels
   - Uses Otsu's method for automatic threshold detection if not specified
   - Returns ternary matrix (0/1/2); saves to CSV via `save_matrix_to_csv()`

### Key Algorithms

**Corner Ordering** (`order_points`)
- Sorts 4 corner points into canonical order: top-left, top-right, bottom-right, bottom-left
- Uses sum and diff of coordinates to determine which corner is which
- Critical for correct perspective transform orientation

**Adaptive Thresholding**
- Uses Gaussian adaptive thresholding to handle non-uniform lighting
- Parameters: block size=11, C=2
- Inverted binary output highlights dark regions (grid lines, black cells)

**Peak Detection**
- Finds grid line positions by detecting peaks in negative projection profiles
- Minimum distance constraint: height/50 or width/50 (prevents noise detection)
- Prominence threshold: width*10 or height*10 (ensures significant lines only)

### Application Class Structure

The `Application` class serves as the main interface:
- `__init__`: Loads image and configures logging (INFO or TRACE level)
- `extract()`: Runs grid extraction, saves straightened image to `extracted_grid.jpg`
- `size()`: Detects and prints grid dimensions
- `convert()`: Full pipeline from image to binary matrix with CSV export

## Output Files

- `extracted_grid.jpg` - Straightened/warped crossword grid image
- `crossword_grid.csv` - Matrix representation with values:
  - `0` = black cell (filled/blocked)
  - `1` = white cell (empty)
  - `2` = white cell with black dot (solution letter location)

## Image Processing Notes

## Recent Improvements

### v0.2.0 - Dot Detection Feature
- **Solution marker detection**: Automatically detects black dots in white cells
- **Ternary output**: Matrix now supports 0 (black), 1 (white), 2 (white with dot)
- **Configurable**: Can be enabled/disabled via `--detect-dots` / `--no-detect-dots` flags
- **Smart detection**: Checks bottom-right corner region for dark pixels
- **MCP integration**: Dot detection available in MCP server tool

### v0.1.1 - CLI Refactoring
- **Replaced Fire with argparse**: CLI now uses Python's built-in argparse module
- **Functional design**: Removed unnecessary class wrapper, using direct function calls
- **Reduced dependencies**: Removed `fire>=0.5.0` from requirements
- **Simpler code**: 275 lines vs ~300, clearer structure
- **Better help**: Professional help messages with usage examples

### v0.1.0 - Core Algorithm Improvements

The codebase has been significantly refactored with the following enhancements:

### Error Handling
- Custom exceptions: `GridExtractionError`, `DimensionDetectionError`
- Proper exception propagation with informative error messages
- Input validation in `Application.__init__` (file existence, valid image format)
- Comprehensive error checking at each pipeline stage

### Configurability
- All magic numbers now exposed as function parameters
- CLI arguments for output paths, thresholds, and visualization
- Auto-threshold detection using Otsu's method (can be overridden)
- Configurable cell margins, contour epsilon, adaptive threshold parameters

### Code Quality
- Full type annotations with proper `tuple[int, int]` syntax
- Comprehensive docstrings for all public functions and classes
- Improved variable names (`point_sum` vs `s`, `perimeter` vs `p`, etc.)
- Consistent use of `logger` instead of mixed `print()` statements

### Features
- Visualization mode: shows detected contours on original image
- Grid statistics: reports white/black cell counts
- Sanity checks: warns about unusually large grid dimensions (>50×50)
- Progress indicators: "Step 1/3..." messages during conversion

### API Changes
- `detect_grid_dimensions()` now returns `(cols, rows)` consistently
- `convert()` returns the grid matrix for programmatic use
- Output paths are configurable (no more hardcoded filenames)
- All functions properly document exceptions they raise

## Important Implementation Details

### Auto-Threshold Detection
The code now uses Otsu's method by default to automatically determine the optimal threshold for black/white classification. This adapts to different:
- Paper colors (cream, white, gray newsprint)
- Print quality (laser, inkjet, newspaper)
- Lighting conditions

Manual threshold can still be specified via `--threshold` when auto-detection fails.

### Cell Sampling Strategy
Uses configurable center crop (default 25% margin on each side) to avoid:
- Grid line pixels contaminating cell classification
- Edge effects from perspective transform interpolation
- Margin is adjustable via `cell_margin` parameter

### Contour Detection
Requires the crossword to be the largest quadrilateral in the image. May fail if:
- Multiple large rectangles are present
- The grid is not the dominant feature
- Heavy distortion prevents quadrilateral approximation

Use `--visualize` flag to debug contour detection issues.

### Dimension Detection Robustness
The median-based approach for dimension detection handles:
- Missing grid lines (faint printing)
- Partially visible grids
- Non-uniform cell sizes (within reason)
- Double-detected grid lines (bimodal distribution refinement)
- Over-detection correction (cross-validation between dimensions)

**Verified Test Cases**:
- 17×12 grid (non-square): Successfully detected with cross-validation
- Handles grids where peak detection initially finds 35 columns → refined to 17
- Handles grids where peak detection initially finds 24-25 rows → refined to 12

However, it requires at least 2 detected lines per axis. If detection fails, try:
- Improving image contrast
- Cropping closer to the grid
- Adjusting `min_distance_factor` and `prominence_factor` parameters

## Testing Considerations

When adding tests, focus on:
- `order_points()` with various corner configurations
- `detect_grid_dimensions()` with known grid sizes and edge cases
- Error handling paths (no contour, invalid dimensions, missing files)
- Different crossword styles (American, British, cryptic)
- Edge cases: very small images, very large grids, missing lines
