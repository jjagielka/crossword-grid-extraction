# Quick Start Guide

## Installation

```bash
# Clone or download the repository
cd crossword-read

# Install dependencies
uv pip install -e .

# For MCP server support
uv pip install -e ".[mcp]"

# For development
uv pip install -e ".[dev]"
```

## Basic CLI Usage

```bash
# Full pipeline: extract, detect, and convert
python src/crossword.py --input=your_crossword.jpg convert

# Just extract and straighten
python src/crossword.py --input=your_crossword.jpg extract

# Just detect dimensions
python src/crossword.py --input=your_crossword.jpg size

# With custom output path
python src/crossword.py --input=your_crossword.jpg convert --output=my_grid.csv

# Enable visualization
python src/crossword.py --input=your_crossword.jpg extract --visualize

# Verbose logging for debugging
python src/crossword.py --input=your_crossword.jpg --verbose convert
```

## MCP Server Usage

### 1. Start the Server

```bash
python src/mcp_server.py
```

### 2. Configure Claude Desktop

Edit your config file:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add:
```json
{
  "mcpServers": {
    "crossword-extractor": {
      "command": "python",
      "args": ["/absolute/path/to/crossword-read/src/mcp_server.py"],
      "env": {}
    }
  }
}
```

### 3. Use with Claude

Example prompts:
- "Can you extract the crossword grid from this image?"
- "What are the dimensions of this crossword?"
- "Convert this crossword to a CSV matrix"

## Python API Usage

```python
import cv2
from crossword import extract_grid, detect_grid_dimensions, convert

# Load image
image = cv2.imread("crossword.jpg")

# Extract and straighten
warped, width, height = extract_grid(image)

# Detect dimensions
cols, rows = detect_grid_dimensions(warped)

# Convert to binary matrix
grid = convert(warped, width, height, rows, cols)

print(f"Grid dimensions: {cols}×{rows}")
print(grid)
```

## Example Files

Test with provided examples:
```bash
# Run all usage examples
python examples/usage_example.py

# Test the MCP server
python tests/test_mcp_server.py

# Run full test suite
pytest -v
```

## Output

The tool produces:
- **extracted_grid.jpg** - Straightened grid image
- **crossword_grid.csv** - Binary matrix (0=black, 1=white)
- Grid dimensions and statistics in the console

## Common Issues

### "Could not detect the grid contour"
→ Crop closer to the grid or improve image contrast

### "Failed to detect valid grid dimensions"
→ Ensure grid lines are visible, try higher quality image

### MCP server not connecting
→ Check absolute path in config, restart Claude Desktop

## Documentation

- **README.md** - Full project documentation
- **docs/MCP_SERVER.md** - MCP server guide
- **CLAUDE.md** - Development guide
- **docs/** - Additional documentation

## Getting Help

- Check troubleshooting sections in README.md
- Run with `--verbose` flag for detailed logs
- Review test examples in `tests/` and `examples/`
