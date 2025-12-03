# Dot Detection Feature

## Overview

The crossword grid extraction tool now supports detecting black dots in white cells that mark solution letter locations. This is a common feature in certain types of crosswords where dots indicate which letters form the final solution.

## How It Works

### Detection Algorithm

1. **Cell Classification**: First, cells are classified as black (0) or white (1) based on average intensity
2. **Dot Check**: For white cells only, the bottom-right corner region is examined
3. **Corner Analysis**: A small region (15% of cell size by default) in the bottom-right is checked
4. **Intensity Comparison**: If the corner region is significantly darker than the threshold, a dot is detected
5. **Output**: Cells with dots are marked as value `2` in the output matrix

### Key Parameters

- **`detect_dots`** (bool, default=True): Enable/disable dot detection
- **`dot_size_ratio`** (float, default=0.15): Size of corner region to check (10-30% of cell)
- **Threshold**: Dot detection uses 70% of the main intensity threshold for stricter dark detection

## Output Format

The matrix now uses ternary values:
- **0** = Black cell (filled/blocked)
- **1** = White cell (empty, no dot)
- **2** = White cell with black dot (solution letter marker)

## Usage

### Command Line

```bash
# Default: dot detection enabled
python src/crossword.py --input puzzle.jpg convert --output grid.csv

# Explicitly enable dot detection
python src/crossword.py --input puzzle.jpg convert --output grid.csv --detect-dots

# Disable dot detection (binary output only)
python src/crossword.py --input puzzle.jpg convert --output grid.csv --no-detect-dots
```

### Python API

```python
from extract import convert_to_matrix

# With dot detection (default)
matrix = convert_to_matrix(image, width, height, rows, cols, detect_dots=True)

# Without dot detection
matrix = convert_to_matrix(image, width, height, rows, cols, detect_dots=False)

# Custom dot size
matrix = convert_to_matrix(
    image, width, height, rows, cols,
    detect_dots=True,
    dot_size_ratio=0.2  # Larger detection region (20%)
)
```

### MCP Server

```python
# Dot detection enabled by default
result = extract_crossword_grid(
    image_base64="...",
    output_format="csv"
)

# Disable dot detection
result = extract_crossword_grid(
    image_base64="...",
    output_format="csv",
    detect_dots=False
)
```

## Example Output

### With Dot Detection (default)
```
Grid statistics: 147 white cells, 53 black cells, 4 cells with dots

Grid Matrix (0=Black, 1=White, 2=White+Dot):
[[1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1]
 [1 1 1 2 1 1 1 1 1 0 1 0 1 0 1 0 1]  <- Cell with dot at column 3
 [2 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1]  <- Cell with dot at column 0
 ...
```

### Without Dot Detection (`--no-detect-dots`)
```
Grid statistics: 151 white cells, 53 black cells

Grid Matrix (0=Black, 1=White):
[[1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1]
 [1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0 1]  <- All white cells are 1
 [1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1 1]
 ...
```

## CSV Output

The CSV file contains the same ternary values:

```csv
1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1
1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1
1,1,1,2,1,1,1,1,1,0,1,0,1,0,1,0,1
2,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1
```

Where `2` indicates cells with detected dots.

## Technical Details

### Dot Detection Logic

The `_detect_dot_in_cell()` function:

1. **Extracts corner region**: Bottom-right `dot_size_ratio` portion of the cell
2. **Avoids edges**: Adds margin to skip grid line pixels
3. **Calculates intensity**: Average pixel intensity in corner region
4. **Applies threshold**: Corner intensity < (main_threshold × 0.7)
5. **Returns boolean**: True if dot detected

### Edge Cases Handled

- **Small cells**: If cell is too small for reliable detection (< 3×3 pixels), returns False
- **Empty corner**: If corner extraction fails, returns False
- **Grid line interference**: Edge margin prevents grid lines from affecting detection
- **Black cells**: Dot detection only runs on white cells (intensity > threshold)

## Configuration

### Adjusting Sensitivity

If dot detection is too sensitive or not sensitive enough:

```python
# More sensitive (smaller region, easier to detect)
matrix = convert_to_matrix(..., dot_size_ratio=0.1)

# Less sensitive (larger region, needs bigger dot)
matrix = convert_to_matrix(..., dot_size_ratio=0.25)
```

### Adjusting Threshold

The dot threshold is automatically 70% of the main intensity threshold. This can be adjusted by modifying the `_detect_dot_in_cell()` function:

```python
# Current: dot_threshold = intensity_threshold * 0.7
# More strict: dot_threshold = intensity_threshold * 0.5
# Less strict: dot_threshold = intensity_threshold * 0.8
```

## Use Cases

1. **Crossword Solutions**: Mark which letters contribute to the final solution phrase
2. **Puzzle Analysis**: Identify solution patterns in crossword databases
3. **Automated Solving**: Focus solving algorithms on solution cells
4. **Grid Validation**: Verify that crosswords follow expected dot placement rules

## Testing

Tested on a 17×12 crossword grid:
- ✅ Detected 4 cells with dots in bottom-right corners
- ✅ No false positives in cells without dots
- ✅ Can be disabled with `--no-detect-dots` flag
- ✅ Works correctly with both CLI and MCP server
- ✅ CSV output correctly shows ternary values (0/1/2)

## Backward Compatibility

The feature is **opt-out by default**:
- Default behavior: `detect_dots=True` (ternary output)
- Legacy behavior: Use `--no-detect-dots` flag (binary output)

This ensures existing users get the enhanced functionality while allowing them to revert to binary output if needed.
