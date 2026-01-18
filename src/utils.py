import cv2
import json
import base64
import numpy as np
from loguru import logger
from pathlib import Path
from typing import Optional, Tuple, Dict

# Import core extraction functions from library
from extract import (
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
)


def load_image(image_path: Path) -> cv2.Mat:
    """Load and validate an image file.

    Args:
        image_path: Path to input crossword image

    Returns:
        Loaded OpenCV image array

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If file is not a valid image
    """
    # Validate input file
    if not image_path.exists():
        raise FileNotFoundError(f"Input file not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Input path is not a file: {image_path}")

    # Load and validate image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(
            f"Failed to load image from {image_path}. "
            f"Ensure the file is a valid image format (JPEG, PNG, etc.)"
        )

    logger.info(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
    return image


def load_image_from_base64(image_base64: str) -> np.ndarray:
    """Decode base64 image data and load with OpenCV.

    Args:
        image_base64: Base64-encoded image data

    Returns:
        Loaded OpenCV image array

    Raises:
        ValueError: If base64 data is invalid or image loading fails
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from base64 data")
        return image
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        raise ValueError(f"Invalid base64 image data: {e}")


def save_matrix_to_csv(grid_matrix: np.ndarray, output_path: Path) -> None:
    """Save grid matrix to CSV file.

    Args:
        grid_matrix: Binary matrix to save
        output_path: Path where CSV should be saved
    """
    np.savetxt(output_path, grid_matrix, fmt="%d", delimiter=",")
    logger.info(f"Saved matrix to {output_path}")


def save_matrix_to_json(grid_matrix: np.ndarray, output_path: Path) -> None:
    """Save grid matrix to JSON file.

    Args:
        grid_matrix: Binary matrix to save
        output_path: Path where JSON should be saved
    """
    with open(output_path, "w") as f:
        json.dump(grid_matrix.tolist(), f)
    logger.info(f"Saved matrix to {output_path}")


def get_grid_stats(grid_matrix: np.ndarray) -> Dict[str, int]:
    """Calculate statistics for a grid matrix.

    Args:
        grid_matrix: Grid matrix (0=black, 1=white, 2=dotted)

    Returns:
        Dictionary with white_cells, black_cells, and dotted_cells counts
    """
    return {
        "white_cells": int(np.sum(grid_matrix == 1)),
        "black_cells": int(np.sum(grid_matrix == 0)),
        "dotted_cells": int(np.sum(grid_matrix == 2)),
    }


def format_matrix(
    grid_matrix: np.ndarray,
    cols: int,
    rows: int,
    output_format: str = "csv",
    detect_dots: bool = True,
) -> str:
    """Format grid matrix as a string in various formats with metadata header.

    Args:
        grid_matrix: Grid matrix to format
        cols: Number of columns
        rows: Number of rows
        output_format: One of "csv", "json", "array"
        detect_dots: Whether dots were detected (for header)

    Returns:
        Formatted string
    """
    stats = get_grid_stats(grid_matrix)

    header = f"Detected: {cols} columns × {rows} rows\n"
    if detect_dots and stats["dotted_cells"] > 0:
        header += f"Grid statistics: {stats['white_cells']} white cells, {stats['black_cells']} black cells, {stats['dotted_cells']} cells with dots\n\n"
    else:
        header += f"Grid statistics: {stats['white_cells']} white cells, {stats['black_cells']} black cells\n\n"

    if output_format == "csv":
        grid_str = "\n".join([",".join(map(str, row)) for row in grid_matrix])
        return header + grid_str
    elif output_format == "json":
        return header + json.dumps(grid_matrix.tolist(), indent=2)
    elif output_format == "array":
        return header + str(grid_matrix)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def process_crossword_image(
    image: np.ndarray,
    intensity_threshold: Optional[int] = None,
    detect_dots: bool = True,
    use_curved_lines: bool = True,
    curve_smoothing: float = 100.0,
    expected_cell_aspect_ratio: float = 1.0,
) -> Tuple[np.ndarray, int, int]:
    """Full pipeline: extract, detect dimensions, and convert to matrix.

    Args:
        image: Input OpenCV image
        intensity_threshold: Optional manual intensity threshold
        detect_dots: Whether to detect dots
        use_curved_lines: Whether to use curved line detection
        curve_smoothing: Smoothing factor for curved lines
        expected_cell_aspect_ratio: Expected cell aspect ratio

    Returns:
        Tuple of (grid_matrix, cols, rows)
    """
    # 1. Extract and straighten grid
    logger.info("Extracting and straightening grid...")
    warped, max_width, max_height = extract_grid(image)
    logger.info(f"Extracted grid: {max_width}x{max_height} pixels")

    # 2. Detect dimensions
    logger.info("Detecting grid dimensions...")
    cols, rows = detect_grid_dimensions(
        warped, expected_cell_aspect_ratio=expected_cell_aspect_ratio
    )
    logger.info(f"Detected: {cols} columns x {rows} rows")

    # 3. Convert to matrix
    logger.info("Converting to matrix...")
    grid_matrix = convert_to_matrix(
        warped,
        max_width,
        max_height,
        rows,
        cols,
        intensity_threshold=intensity_threshold,
        detect_dots=detect_dots,
        use_curved_lines=use_curved_lines,
        curve_smoothing=curve_smoothing,
    )

    return grid_matrix, cols, rows
