# Library Usage Guide

The crossword grid extraction functionality has been organized into a reusable library that can be used by multiple interfaces.

## Architecture

```
src/
├── extract.py       # Core image processing library (reusable)
├── crossword.py     # CLI interface (uses extract.py)
└── mcp_server.py    # MCP server interface (uses extract.py)
```

## Using the Library

### Python API

You can import and use the core functions directly in your Python code:

```python
import cv2
from extract import (
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
    save_matrix_to_csv,
    GridExtractionError,
    DimensionDetectionError,
)

# Load image
img = cv2.imread("crossword.jpg")

# Extract and straighten grid
warped, width, height = extract_grid(img)

# Detect dimensions
cols, rows = detect_grid_dimensions(warped)

# Convert to binary matrix
grid_matrix = convert_to_matrix(warped, width, height, rows, cols)

# Save to CSV (optional)
from pathlib import Path
save_matrix_to_csv(grid_matrix, Path("output.csv"))
```

### Available Functions

#### `extract_grid(img, contour_epsilon=0.02, adaptive_block_size=11, adaptive_c=2)`
Extracts and straightens the crossword grid from an image.

**Parameters:**
- `img` (np.ndarray): Input image in BGR format
- `contour_epsilon` (float): Approximation accuracy for contour detection
- `adaptive_block_size` (int): Block size for adaptive thresholding
- `adaptive_c` (int): Constant for adaptive threshold

**Returns:**
- `(warped_image, width, height)`: Straightened grid and its dimensions

**Raises:**
- `GridExtractionError`: If grid cannot be detected

---

#### `detect_grid_dimensions(img, min_distance_factor=50, prominence_factor=10)`
Detects the number of rows and columns in the grid.

**Parameters:**
- `img` (np.ndarray): Straightened grid image (BGR or grayscale)
- `min_distance_factor` (int): Divisor for minimum peak distance
- `prominence_factor` (int): Multiplier for peak prominence

**Returns:**
- `(cols, rows)`: Number of columns and rows

**Raises:**
- `DimensionDetectionError`: If dimensions cannot be detected

---

#### `convert_to_matrix(image, max_width, max_height, rows, cols, intensity_threshold=None, cell_margin=0.25)`
Converts the straightened grid to a binary matrix.

**Parameters:**
- `image` (np.ndarray): Straightened grid image
- `max_width` (int): Width in pixels
- `max_height` (int): Height in pixels
- `rows` (int): Number of rows
- `cols` (int): Number of columns
- `intensity_threshold` (int, optional): Threshold for black/white classification (auto-detected if None)
- `cell_margin` (float): Fraction of cell to crop from edges (0.0-0.5)

**Returns:**
- `np.ndarray`: Binary matrix where 0=black cell, 1=white cell

**Raises:**
- `ValueError`: If dimensions are invalid

---

#### `save_matrix_to_csv(grid_matrix, output_path)`
Saves a grid matrix to a CSV file.

**Parameters:**
- `grid_matrix` (np.ndarray): Binary matrix to save
- `output_path` (Path): Where to save the CSV

---

## Exception Handling

```python
try:
    warped, width, height = extract_grid(img)
except GridExtractionError as e:
    print(f"Could not find crossword grid: {e}")
    # Handle error...

try:
    cols, rows = detect_grid_dimensions(warped)
except DimensionDetectionError as e:
    print(f"Could not detect dimensions: {e}")
    # Handle error...
```

## Example: Full Pipeline

See `example_usage.py` for a complete working example.

```python
#!/usr/bin/env python3
import cv2
from pathlib import Path
from extract import (
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
    save_matrix_to_csv,
)

# Load image
img = cv2.imread("crossword.jpg")

# Process
warped, width, height = extract_grid(img)
cols, rows = detect_grid_dimensions(warped)
matrix = convert_to_matrix(warped, width, height, rows, cols)

# Save
save_matrix_to_csv(matrix, Path("output.csv"))

# Display
print(f"Grid: {cols}x{rows}")
print(matrix)
```

## Integration Examples

### CLI Interface (`crossword.py`)
The CLI uses Fire to expose library functions as commands:
```python
from extract import extract_grid, detect_grid_dimensions, convert_to_matrix
# ... orchestrate functions based on CLI commands
```

### MCP Server (`mcp_server.py`)
The MCP server wraps library functions as MCP tools:
```python
from extract import extract_grid, detect_grid_dimensions, convert_to_matrix

@mcp.tool()
def extract_crossword_grid(image_base64: str) -> str:
    # Use library functions to process base64 image
    # Return formatted results
```

## Algorithm Details

### Dimension Detection
The dimension detection algorithm includes several advanced features:

1. **Bimodal Distribution Refinement**: Detects when grid lines are double-detected (both edges detected separately) and filters duplicates
2. **Cross-Validation**: If one dimension is detected successfully but the other shows suspicious results (>30 cells), uses the good dimension to validate the other
3. **Median-based Estimation**: Robust to missing or faint grid lines

### Threshold Detection
The matrix conversion uses Otsu's method for automatic threshold detection, adapting to:
- Different paper colors (white, cream, newsprint)
- Print quality variations (laser, inkjet, newspaper)
- Lighting conditions

### Cell Sampling
Uses configurable center cropping (default 25% margin on each side) to avoid:
- Grid line pixels contaminating cell classification
- Edge effects from perspective transform interpolation
