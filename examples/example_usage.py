#!/usr/bin/env python3
"""Example usage of the extract.py library.

This demonstrates how to use the crossword grid extraction library
programmatically, without using the CLI interface.
"""

import cv2
from pathlib import Path
import sys

# Add src to path if running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from extract import (
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
    save_matrix_to_csv,
    GridExtractionError,
    DimensionDetectionError,
)
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")


def process_crossword(image_path: str, output_csv: str) -> None:
    """Process a crossword image end-to-end.

    Args:
        image_path: Path to input crossword image
        output_csv: Path where CSV output should be saved
    """
    try:
        # 1. Load image
        logger.info(f"Loading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 2. Extract and straighten grid
        logger.info("Extracting grid...")
        warped, width, height = extract_grid(img)
        logger.info(f"Extracted grid: {width}x{height} pixels")

        # 3. Detect grid dimensions
        logger.info("Detecting dimensions...")
        cols, rows = detect_grid_dimensions(warped)
        logger.info(f"Detected: {cols} columns x {rows} rows")

        # 4. Convert to binary matrix
        logger.info("Converting to binary matrix...")
        grid_matrix = convert_to_matrix(
            warped,
            width,
            height,
            rows,
            cols,
            intensity_threshold=None,  # Auto-detect
        )

        # 5. Save to CSV
        save_matrix_to_csv(grid_matrix, Path(output_csv))
        logger.info(f"Saved to: {output_csv}")

        # 6. Display matrix
        print("\nGrid Matrix (0=Black, 1=White):")
        print(grid_matrix)

    except (GridExtractionError, DimensionDetectionError) as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    input_image = "path/to/crossword.jpg"
    output_file = "output.csv"

    process_crossword(input_image, output_file)
