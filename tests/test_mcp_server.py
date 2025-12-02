#!/usr/bin/env python3
"""Test script for the MCP server."""

import base64
import sys
from pathlib import Path

# Add src directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test the underlying functions directly, not via MCP
import cv2
import numpy as np
from crossword import extract_grid, detect_grid_dimensions, convert


def test_grid_info():
    """Test the get_grid_info functionality."""
    print("=" * 80)
    print("Testing grid info extraction...")
    print("=" * 80)

    # Load test image
    image_path = Path("test_data/nyt1.png")
    image = cv2.imread(str(image_path))

    orig_height, orig_width = image.shape[:2]

    # Extract grid
    warped, max_width, max_height = extract_grid(image)

    # Detect dimensions
    cols, rows = detect_grid_dimensions(warped)

    # Calculate cell size
    cell_width = max_width / cols
    cell_height = max_height / rows

    result = f"Image size: {orig_width}×{orig_height} pixels\n"
    result += f"Extracted grid: {max_width}×{max_height} pixels\n"
    result += f"Detected dimensions: {cols} columns × {rows} rows\n"
    result += f"Estimated cell size: {cell_width:.1f}×{cell_height:.1f} pixels\n"

    print(result)
    print()


def test_extract_crossword_grid():
    """Test the extract_crossword_grid functionality."""
    print("=" * 80)
    print("Testing crossword grid extraction (CSV format)...")
    print("=" * 80)

    # Load test image
    image_path = Path("test_data/nyt1.png")
    image = cv2.imread(str(image_path))

    # Extract and straighten grid
    warped, max_width, max_height = extract_grid(image)

    # Detect dimensions
    cols, rows = detect_grid_dimensions(warped)

    # Convert to binary matrix
    grid_matrix = convert(warped, max_width, max_height, rows, cols, output_path=None)

    # Format as CSV
    white_cells = int(np.sum(grid_matrix == 1))
    black_cells = int(np.sum(grid_matrix == 0))

    header = f"Detected: {cols} columns × {rows} rows\n"
    header += f"Grid statistics: {white_cells} white cells, {black_cells} black cells\n\n"
    grid_str = "\n".join([",".join(map(str, row)) for row in grid_matrix])
    result = header + grid_str

    # Print first 20 lines
    lines = result.split("\n")
    print("\n".join(lines[:20]))
    if len(lines) > 20:
        print(f"... ({len(lines) - 20} more lines)")
    print()


def test_extract_json_format():
    """Test the extract_crossword_grid functionality with JSON format."""
    print("=" * 80)
    print("Testing crossword grid extraction (JSON format)...")
    print("=" * 80)

    # Load test image
    image_path = Path("test_data/nyt1.png")
    image = cv2.imread(str(image_path))

    # Extract and straighten grid
    warped, max_width, max_height = extract_grid(image)

    # Detect dimensions
    cols, rows = detect_grid_dimensions(warped)

    # Convert to binary matrix
    grid_matrix = convert(warped, max_width, max_height, rows, cols, output_path=None)

    # Format as JSON
    import json
    white_cells = int(np.sum(grid_matrix == 1))
    black_cells = int(np.sum(grid_matrix == 0))

    header = f"Detected: {cols} columns × {rows} rows\n"
    header += f"Grid statistics: {white_cells} white cells, {black_cells} black cells\n\n"
    result = header + json.dumps(grid_matrix.tolist(), indent=2)

    # Print first 30 lines
    lines = result.split("\n")
    print("\n".join(lines[:30]))
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    print()


if __name__ == "__main__":
    try:
        test_grid_info()
        test_extract_crossword_grid()
        test_extract_json_format()
        print("=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
