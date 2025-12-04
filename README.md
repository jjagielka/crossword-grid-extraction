# Crossword-Read

A Python computer vision tool for extracting crossword grid structures from images. Uses OpenCV to detect, straighten, and digitize crossword puzzles into binary matrices.

## Features

- **Automatic Grid Detection**: Finds and extracts crossword grids from photos
- **Perspective Correction**: Straightens skewed or angled images
- **Dimension Detection**: Automatically determines grid size (rows x columns)
- **Dot Detection**: Identifies black dots marking solution letter positions (NEW in v0.2.0)
- **Ternary Matrix Export**: Converts grids to CSV format (0=black, 1=white, 2=white with dot)
- **Auto-Thresholding**: Uses Otsu's method for adaptive cell classification
- **Bimodal Distribution Handling**: Robust detection even with thick grid lines
- **Visualization Mode**: Shows detected contours and processing steps
- **Configurable Parameters**: Adjust thresholds and detection settings via CLI
- **MCP Server**: Integrate with LLMs via Model Context Protocol (see [docs/MCP_SERVER.md](docs/MCP_SERVER.md))

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

### Basic Commands

```bash
# Extract and straighten the grid
python src/crossword.py --input=crossword.jpg extract

# Detect grid dimensions only
python src/crossword.py --input=crossword.jpg size

# Full pipeline: extract, detect, and convert to CSV
python src/crossword.py --input=crossword.jpg convert
```

### Advanced Options

```bash
# Custom output path
python src/crossword.py --input=crossword.jpg extract --output=my_grid.jpg

# Enable visualization (shows detected contours)
python src/crossword.py --input=crossword.jpg extract --visualize

# Convert with custom intensity threshold
python src/crossword.py --input=crossword.jpg convert --threshold=150

# Convert with custom output path
python src/crossword.py --input=crossword.jpg convert --output=grid.csv

# Enable verbose logging for debugging
python src/crossword.py --input=crossword.jpg --verbose convert

# Disable dot detection (output binary matrix only)
python src/crossword.py --input=crossword.jpg convert --no-detect-dots
```

## How It Works

1. **Grid Extraction**:
   - Applies adaptive thresholding to handle lighting variations
   - Detects the largest quadrilateral contour (the crossword boundary)
   - Performs perspective transform to straighten the grid

2. **Dimension Detection**:
   - Uses projection profiles (pixel intensity sums along axes)
   - Applies peak detection to find grid lines
   - Detects and corrects bimodal distributions (double-detected thick lines)
   - Calculates median cell size for robustness against missing lines
   - Cross-validates dimensions using cell aspect ratio

3. **Grid Conversion**:
   - Slices the straightened image into individual cells
   - Samples the center region of each cell (avoiding grid lines)
   - Classifies cells as black (0) or white (1) using auto-detected threshold
   - Optionally detects black dots in bottom-right corner (2=white with dot)
   - Uses relative intensity comparison for robust dot detection

## Output

By default, output files are saved to the current directory (or use `--output` to specify):
- **extracted_grid.jpg**: Straightened grid image
- **crossword_grid.csv**: Ternary matrix with values:
  - `0` = black cell (filled)
  - `1` = white cell (empty)
  - `2` = white cell with black dot (solution letter position)
- **Visualization images** (when `--visualize` is used)

Output files are automatically created in the `output/` folder when using default paths.

### Example Output

For a 17×12 crossword grid with 10 solution dots:
```
Detected: 17 columns × 12 rows
Grid statistics: 141 white cells, 53 black cells, 10 cells with dots

1,0,1,0,1,0,1,0,1,1,2,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,2
...
```

## Project Structure

```
crossword-grid-extraction/
├── README.md             # This file
├── CLAUDE.md             # Development guide for Claude Code
├── pyproject.toml        # Project configuration and dependencies
│
├── src/                  # Source code
│   ├── extract.py        # Core CV library (shared)
│   ├── crossword.py      # CLI interface (argparse)
│   └── mcp_server.py     # MCP server interface
│
├── docs/                 # Documentation
│   ├── MCP_SERVER.md                    # MCP server setup and usage
│   ├── LIBRARY_USAGE.md                 # Library API documentation
│   ├── DOT_DETECTION_FEATURE.md         # Dot detection feature guide
│   ├── SHADOW_HANDLING.md               # Shadow/lighting handling
│   ├── ASPECT_RATIO_VALIDATION.md       # Validation approach
│   └── ...
│
├── examples/             # Usage examples
│   ├── example_usage.py  # Programmatic library usage
│   └── ...
│
├── test_data/            # Test images for validation
│   ├── wciska_kig.jpg    # Normal lighting (17×12)
│   ├── zcieniem.jpg      # With shadows (17×12)
│   ├── crosswords2.jpg   # Different grid (17×12)
│   ├── jolka.jpg         # Portrait format (11×19)
│   └── ...
│
├── dev/                  # Development files (debug/test scripts)
│   ├── README.md         # Dev folder documentation
│   ├── debug_*.py        # Diagnostic scripts
│   ├── test_*.py         # Test scripts
│   └── ...
│
├── output/               # Generated output files (gitignored)
│   ├── extracted_grid.jpg
│   └── crossword_grid.csv
│
└── tests/                # Formal test suite
```

## Requirements

- Python 3.11+
- OpenCV (opencv-python) >= 4.8.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Loguru >= 0.7.0 (logging)

Optional for MCP server:
- FastMCP >= 0.1.0 (MCP server framework)

## Limitations

- Requires the crossword to be the largest quadrilateral object in the image
- Works best with clear, well-lit images
- Grid lines must be reasonably visible for dimension detection
- Typical crossword sizes (10-25 cells) work best

## MCP Server (LLM Integration)

This project includes an MCP (Model Context Protocol) server for integration with LLMs like Claude.

```bash
# Install MCP dependencies
uv pip install fastmcp

# Run the MCP server (STDIO mode for Claude Desktop)
python src/mcp_server.py

# Run as HTTP service (for persistent deployment)
python src/mcp_server.py --http

# Test the MCP server
python tests/test_mcp_server.py
```

**Documentation:**
- **[docs/MCP_SERVER.md](docs/MCP_SERVER.md)** - Complete MCP server guide
  - Available tools (extract_crossword_grid, get_grid_info)
  - Configuration for Claude Desktop and other MCP clients
  - Usage examples, API reference, and troubleshooting
- **[docs/DEPLOYMENT_OPTIONS.md](docs/DEPLOYMENT_OPTIONS.md)** - Choosing between STDIO and HTTP/SSE modes
- **[docs/SYSTEMD_DEPLOYMENT.md](docs/SYSTEMD_DEPLOYMENT.md)** - Running as a systemd service on Linux

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Type checking (when available)
mypy src/

# Linting (when available)
ruff check src/

# Format check (when available)
ruff format --check src/
```

## Recent Changes (v0.2.0)

### Dot Detection Feature
- **NEW**: Detects black dots in white cells marking solution letter positions
- Ternary output: 0 (black), 1 (white), 2 (white with dot)
- 100% accuracy on test images with configurable sensitivity
- Uses relative intensity comparison for robust detection
- CLI flag: `--detect-dots` (enabled by default), `--no-detect-dots` to disable
- MCP server parameter: `detect_dots=True/False`

### Improved Dimension Detection
- **FIXED**: Column duplication issue with thick grid lines
- **NEW**: Bimodal distribution detection for both rows and columns
- More robust handling of double-detected grid line edges
- Cross-validation between dimensions for accuracy
- Handles various grid line thicknesses and styles

### Architecture Improvements
- **REFACTORED**: Extracted core CV logic into `src/extract.py` library
- **REPLACED**: Fire CLI framework with standard argparse (no external CLI deps)
- **SIMPLIFIED**: Functional CLI approach instead of class-based
- Shared library between CLI and MCP server interfaces

## Troubleshooting

**"Could not detect the grid contour"**
- Ensure the crossword is the largest object in the image
- Try improving image contrast or cropping closer to the grid
- Use `--visualize` to see what contours are detected

**"Failed to detect valid grid dimensions"**
- Grid lines may be too faint
- Try adjusting the image or using `--threshold` parameter
- Enable `--verbose` to see detection details and bimodal distribution analysis

**Duplicate rows or columns detected**
- Fixed in v0.2.0 with bimodal distribution detection
- Enable `--verbose` to see refinement messages
- The algorithm now automatically corrects double-detected grid lines

**Poor cell classification**
- Use `--threshold` to manually set the black/white threshold
- Default auto-detection (Otsu's method) works for most cases
- Try `--visualize` to inspect the extraction quality

**False positive or missed dot detection**
- Dots must be in the bottom-right corner of white cells
- At least 11% darker than the cell center to be detected
- Use `--no-detect-dots` if you don't need solution markers

## License

MIT License
