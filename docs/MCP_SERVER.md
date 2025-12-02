# Crossword MCP Server

An MCP (Model Context Protocol) server that exposes crossword grid extraction functionality to LLMs.

## Overview

This server provides two tools that allow LLMs to extract and analyze crossword puzzles from images:

1. **extract_crossword_grid** - Full extraction pipeline (extract, detect, convert to matrix)
2. **get_grid_info** - Quick grid analysis (extract and detect dimensions only)

## Installation

```bash
# Install dependencies
uv pip install -e .
uv pip install fastmcp

# Or using pip
pip install -e .
pip install fastmcp
```

## Running the Server

The server supports two transport modes:

### STDIO Mode (Default - for Claude Desktop)

```bash
python src/mcp_server.py
```

The server will start in STDIO mode, ready to receive MCP protocol messages via standard input/output. This mode is used by Claude Desktop and similar desktop MCP clients.

### HTTP/SSE Mode (for persistent deployment)

```bash
python src/mcp_server.py --http
```

The server will start as an HTTP service on port 8000 with Server-Sent Events (SSE) transport. This mode is suitable for:
- Systemd service deployment (see `docs/SYSTEMD_DEPLOYMENT.md`)
- Web-based MCP clients
- Remote MCP access

Server URL: `http://127.0.0.1:8000/sse`

FastMCP includes uvicorn internally, so no additional server is needed.

### Configuration for Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "crossword-extractor": {
      "command": "python",
      "args": ["/path/to/crossword-read/src/mcp_server.py"],
      "env": {}
    }
  }
}
```

### Configuration for Other MCP Clients

The server uses STDIO transport and is compatible with any MCP client. Configure according to your client's documentation.

## Available Tools

### 1. extract_crossword_grid

Performs full crossword grid extraction and conversion.

**Parameters:**
- `image_base64` (string, required): Base64-encoded image data (JPEG, PNG, etc.)
- `output_format` (string, optional): Output format - "csv" (default), "array", or "json"
- `intensity_threshold` (integer, optional): Manual threshold for black/white classification (auto-detected if not provided)

**Returns:**
String containing the grid matrix with metadata:
- Detected dimensions (columns × rows)
- Grid statistics (white cells, black cells)
- Binary matrix in requested format (0=black cell, 1=white cell)

**Example Output (CSV format):**
```
Detected: 17 columns × 12 rows
Grid statistics: 150 white cells, 54 black cells

0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0
0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1
...
```

**Example Output (JSON format):**
```json
Detected: 17 columns × 12 rows
Grid statistics: 150 white cells, 54 black cells

[
  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
  ...
]
```

**Errors:**
- `ValueError`: Invalid image data, unsupported format, or processing failure
- Includes detailed troubleshooting tips in error messages

### 2. get_grid_info

Quick analysis to get grid information without full conversion.

**Parameters:**
- `image_base64` (string, required): Base64-encoded image data (JPEG, PNG, etc.)

**Returns:**
String with grid information:
- Original image size
- Extracted grid size
- Detected dimensions
- Estimated cell size

**Example Output:**
```
Image size: 1536×2048 pixels
Extracted grid: 1078×771 pixels
Detected dimensions: 17 columns × 12 rows
Estimated cell size: 63.4×64.2 pixels
```

**Use Case:**
Useful for quickly checking if a crossword image can be processed before performing full extraction.

## Usage Examples

### Python Script (for testing)

```python
import base64
from pathlib import Path

# Read and encode image
image_path = Path("crossword.jpg")
with open(image_path, "rb") as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode("utf-8")

# The MCP server would receive this base64 string
# and call the appropriate tool function
```

### LLM Usage

When using an LLM with MCP support (like Claude Desktop with MCP enabled):

```
User: Extract the crossword grid from this image [attaches crossword.jpg]

LLM: I'll use the crossword extraction tool to analyze this image.
[Calls extract_crossword_grid tool with base64-encoded image]

Result: Detected a 17×12 crossword grid with 150 white cells and 54 black cells.
Here's the binary matrix: [displays the grid]
```

## How It Works

The server performs these steps:

1. **Grid Extraction**:
   - Applies adaptive thresholding to handle lighting variations
   - Detects the largest quadrilateral contour (the crossword boundary)
   - Performs perspective transform to straighten the grid

2. **Dimension Detection**:
   - Uses projection profiles (pixel intensity sums along axes)
   - Applies peak detection to find grid lines
   - Implements bimodal distribution detection to filter noise
   - Calculates median cell size for robustness

3. **Grid Conversion**:
   - Slices the straightened image into individual cells
   - Samples the center region of each cell (avoiding grid lines)
   - Classifies cells as black (0) or white (1) using auto-detected threshold (Otsu's method)

## Requirements

- Python 3.11+
- OpenCV (opencv-python)
- NumPy
- SciPy
- Loguru (logging)
- FastMCP (MCP server framework)

## Limitations

- Requires the crossword to be the largest quadrilateral object in the image
- Works best with clear, well-lit images
- Grid lines must be reasonably visible for dimension detection
- Typical crossword sizes (10-25 cells) work best
- Very large grids (>50 cells) may trigger warnings

## Troubleshooting

### "Could not detect the grid contour"
- Ensure the crossword is the largest object in the image
- Try improving image contrast or cropping closer to the grid
- Make sure the grid edges are clearly visible

### "Failed to detect valid grid dimensions"
- Grid lines may be too faint or irregular
- Try using a higher quality image
- Ensure grid lines are reasonably visible
- Enable verbose logging for detailed diagnostics

### Poor cell classification
- Use `intensity_threshold` parameter to manually set the black/white threshold
- Default auto-detection (Otsu's method) works for most cases
- Try different threshold values (typical range: 100-200)

## Testing

Run the test script to verify functionality:

```bash
python tests/test_mcp_server.py
```

This will:
1. Test grid info extraction
2. Test full conversion with CSV format
3. Test full conversion with JSON format

## Architecture

```
┌─────────────────────┐
│   LLM Client        │
│ (Claude Desktop)    │
└──────────┬──────────┘
           │ MCP Protocol (STDIO)
           │
┌──────────▼──────────┐
│   MCP Server        │
│ (mcp_server.py)     │
├─────────────────────┤
│ Tools:              │
│ - extract_crossword │
│ - get_grid_info     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Core Functions      │
│ (crossword.py)      │
├─────────────────────┤
│ - extract_grid()    │
│ - detect_grid_dims()│
│ - convert()         │
└─────────────────────┘
           │
┌──────────▼──────────┐
│   OpenCV + NumPy    │
│ Image Processing    │
└─────────────────────┘
```

## License

MIT License (same as the main project)

## Related Documentation

- Main project: `README.md`
- Development guide: `CLAUDE.md`
- Test documentation: `tests/README.md`
- Changelog: `docs/CHANGELOG.md`
