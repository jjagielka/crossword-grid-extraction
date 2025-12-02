#!/usr/bin/env python3
"""Example usage of the crossword extraction library.

This script demonstrates how to use the crossword extraction functions
programmatically in Python, rather than via the CLI.
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

# Add src directory to path to import crossword module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crossword import extract_grid, detect_grid_dimensions, convert


def example_basic_usage():
    """Basic usage: extract and convert a crossword grid."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Load image
    image_path = Path("test_data/crossword1.jpg")
    image = cv2.imread(str(image_path))

    # Step 1: Extract and straighten the grid
    print("Step 1: Extracting grid...")
    warped, max_width, max_height = extract_grid(image)
    print(f"  ✓ Extracted grid: {max_width}×{max_height} pixels")

    # Step 2: Detect dimensions
    print("Step 2: Detecting dimensions...")
    cols, rows = detect_grid_dimensions(warped)
    print(f"  ✓ Detected: {cols} columns × {rows} rows")

    # Step 3: Convert to binary matrix
    print("Step 3: Converting to binary matrix...")
    grid_matrix = convert(warped, max_width, max_height, rows, cols, output_path=None)
    print(f"  ✓ Matrix shape: {grid_matrix.shape}")

    # Display statistics
    white_cells = int(np.sum(grid_matrix == 1))
    black_cells = int(np.sum(grid_matrix == 0))
    print(f"\nStatistics:")
    print(f"  White cells: {white_cells}")
    print(f"  Black cells: {black_cells}")

    print("\nFirst 5 rows of grid matrix:")
    print(grid_matrix[:5])
    print()


def example_save_outputs():
    """Example: Save all intermediate outputs."""
    print("=" * 80)
    print("Example 2: Save All Outputs")
    print("=" * 80)

    # Load image
    image_path = Path("test_data/crossword2.jpg")
    image = cv2.imread(str(image_path))

    # Extract grid and save
    print("Extracting and saving grid...")
    warped, max_width, max_height = extract_grid(image)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / "extracted.jpg"), warped)
    print(f"  ✓ Saved to: {output_dir / 'extracted.jpg'}")

    # Detect and convert
    cols, rows = detect_grid_dimensions(warped)
    grid_matrix = convert(
        warped,
        max_width,
        max_height,
        rows,
        cols,
        output_path=output_dir / "grid.csv",
    )
    print(f"  ✓ Saved to: {output_dir / 'grid.csv'}")
    print()


def example_custom_threshold():
    """Example: Using custom intensity threshold."""
    print("=" * 80)
    print("Example 3: Custom Threshold")
    print("=" * 80)

    # Load image
    image_path = Path("test_data/crossword1.jpg")
    image = cv2.imread(str(image_path))

    # Extract and detect
    warped, max_width, max_height = extract_grid(image)
    cols, rows = detect_grid_dimensions(warped)

    # Try different thresholds
    thresholds = [100, 130, 160]
    print("Comparing different thresholds:")

    for threshold in thresholds:
        grid_matrix = convert(
            warped,
            max_width,
            max_height,
            rows,
            cols,
            output_path=None,
            intensity_threshold=threshold,
        )
        white_cells = int(np.sum(grid_matrix == 1))
        black_cells = int(np.sum(grid_matrix == 0))
        print(f"  Threshold {threshold:3d}: {white_cells:3d} white, {black_cells:3d} black")

    print()


def example_error_handling():
    """Example: Error handling."""
    print("=" * 80)
    print("Example 4: Error Handling")
    print("=" * 80)

    from crossword import GridExtractionError, DimensionDetectionError

    # Try to load non-existent image
    print("Attempting to load non-existent image...")
    try:
        image = cv2.imread("nonexistent.jpg")
        if image is None:
            raise ValueError("Failed to load image")
        warped, max_width, max_height = extract_grid(image)
    except (ValueError, GridExtractionError) as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")

    # Try with invalid image (all white)
    print("\nAttempting to detect grid in blank image...")
    try:
        blank_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        warped, max_width, max_height = extract_grid(blank_image)
    except GridExtractionError as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")
        print(f"     Message: {str(e)[:60]}...")

    print()


def example_base64_encoding():
    """Example: Encoding image as base64 (for MCP server)."""
    print("=" * 80)
    print("Example 5: Base64 Encoding (for MCP)")
    print("=" * 80)

    # Load image and convert to base64
    image_path = Path("test_data/crossword1.jpg")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    print(f"Original file size: {len(image_bytes):,} bytes")
    print(f"Base64 string length: {len(image_base64):,} characters")
    print(f"First 100 characters: {image_base64[:100]}...")
    print("\nThis base64 string can be sent to the MCP server's extract_crossword_grid tool.")
    print()


if __name__ == "__main__":
    # Run all examples
    try:
        example_basic_usage()
        example_save_outputs()
        example_custom_threshold()
        example_error_handling()
        example_base64_encoding()

        print("=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
