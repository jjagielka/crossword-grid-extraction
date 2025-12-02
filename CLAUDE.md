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
The application uses `fire` for CLI and can be invoked directly:

```bash
# Extract and straighten the grid
python src/crossword.py --input=crosswords1.jpg extract --output=grid.jpg

# Detect grid dimensions (columns x rows)
python src/crossword.py --input=crosswords1.jpg size

# Full conversion: extract, detect, convert to CSV
python src/crossword.py --input=crosswords1.jpg convert --output=grid.csv

# Use custom intensity threshold
python src/crossword.py --input=crosswords1.jpg convert --threshold=150

# Enable visualization mode (shows detected contours)
python src/crossword.py --input=crosswords1.jpg extract --visualize

# Enable verbose logging (TRACE level)
python src/crossword.py --input=crosswords1.jpg --verbose convert
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
- `fire>=0.5.0` - CLI interface
- `loguru>=0.7.0` - Structured logging

## Architecture

### Processing Pipeline

The application follows a three-stage pipeline:

1. **Grid Extraction** (`extract_grid`)
   - Applies adaptive thresholding to handle shadows/lighting variations
   - Detects the largest quadrilateral contour (the crossword boundary)
   - Performs perspective transform to straighten the grid
   - Returns warped image with computed dimensions

2. **Dimension Detection** (`detect_grid_dimensions`)
   - Uses projection profiles (sum of pixel intensities along axes)
   - Applies peak detection on inverted projections to find grid lines
   - Calculates median cell size from detected peaks
   - Estimates rows/columns by dividing image dimensions by median cell size
   - Robust to missing or faint grid lines through median-based approach

3. **Grid Conversion** (`convert`)
   - Slices straightened image into individual cells based on detected dimensions
   - Samples center region of each cell (50% of cell area) to avoid grid line interference
   - Classifies cells as black (0) or white (1) based on average intensity threshold (140)
   - Outputs binary matrix and optionally saves to CSV

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
- `crossword_grid.csv` - Binary matrix representation (when `save=True`)

## Image Processing Notes

## Recent Improvements (v0.1.0)

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
