# Crossword-Read

A Python computer vision tool for extracting crossword grid structures from images. Uses OpenCV to detect, straighten, and digitize crossword puzzles into binary matrices.

## Features

- **Automatic Grid Detection**: Finds and extracts crossword grids from photos
- **Perspective Correction**: Straightens skewed or angled images
- **Dimension Detection**: Automatically determines grid size (rows x columns)
- **Binary Matrix Export**: Converts grids to CSV format (0=black, 1=white)
- **Auto-Thresholding**: Uses Otsu's method for adaptive cell classification
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
```

## How It Works

1. **Grid Extraction**:
   - Applies adaptive thresholding to handle lighting variations
   - Detects the largest quadrilateral contour (the crossword boundary)
   - Performs perspective transform to straighten the grid

2. **Dimension Detection**:
   - Uses projection profiles (pixel intensity sums along axes)
   - Applies peak detection to find grid lines
   - Calculates median cell size for robustness against missing lines

3. **Grid Conversion**:
   - Slices the straightened image into individual cells
   - Samples the center region of each cell (avoiding grid lines)
   - Classifies cells as black (0) or white (1) using auto-detected threshold

## Output

By default, output files are saved to the current directory (or use `--output` to specify):
- **extracted_grid.jpg**: Straightened grid image
- **crossword_grid.csv**: Binary matrix (0=black, 1=white cells)
- **Visualization images** (when `--visualize` is used)

Output files are automatically created in the `output/` folder when using default paths.

## Project Structure

```
crossword-read/
├── README.md             # This file
├── CLAUDE.md             # Development guide
├── pyproject.toml        # Project configuration
├── pytest.ini            # Test configuration
│
├── src/                  # Source code
│   ├── crossword.py      # Main application
│   └── mcp_server.py     # MCP server
│
├── docs/                 # Documentation
│   ├── MCP_SERVER.md
│   ├── QUICK_START.md
│   ├── QUICK_START_SYSTEMD.md
│   ├── DEPLOYMENT_OPTIONS.md
│   ├── SYSTEMD_DEPLOYMENT.md
│   └── ...
│
├── test_data/            # Test images and expected results
│   ├── crossword1.jpg
│   ├── crossword2.jpg
│   ├── result1.txt
│   └── result2.txt
│
├── tests/                # Test suite
│   ├── conftest.py       # Pytest fixtures
│   ├── test_crossword.py # Main tests
│   └── README.md
│
├── output/               # Generated output files (gitignored)
│   ├── extracted_grid.jpg
│   └── crossword_grid.csv
│
└── examples/             # Example scripts
```

## Requirements

- Python 3.11+
- OpenCV (opencv-python)
- NumPy
- SciPy
- Fire (CLI framework)
- Loguru (logging)

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

# Run the MCP server
python src/mcp_server.py

# Test the MCP server
python tests/test_mcp_server.py
```

See [docs/MCP_SERVER.md](docs/MCP_SERVER.md) for detailed documentation on:
- Configuring the server with Claude Desktop or other MCP clients
- Available tools (extract_crossword_grid, get_grid_info)
- Usage examples and API reference

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest --cov=crossword    # With coverage

# Type checking
mypy src/crossword.py

# Linting
ruff check src/crossword.py
```

## Testing

The project includes a comprehensive test suite with 35 tests covering:
- Integration tests for full pipeline
- Unit tests for all core functions
- MCP server functionality tests
- Error handling and edge cases
- Input validation
- Output format consistency
- Configurable parameters

Run `pytest -v` to see all test results. See `tests/README.md` for detailed testing documentation.

## Troubleshooting

**"Could not detect the grid contour"**
- Ensure the crossword is the largest object in the image
- Try improving image contrast or cropping closer to the grid

**"Failed to detect valid grid dimensions"**
- Grid lines may be too faint
- Try adjusting the image or using `--threshold` parameter
- Enable `--verbose` to see detection details

**Poor cell classification**
- Use `--threshold` to manually set the black/white threshold
- Default auto-detection (Otsu's method) works for most cases
- Try `--visualize` to inspect the extraction quality

## License

MIT License
