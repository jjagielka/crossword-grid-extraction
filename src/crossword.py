#!/usr/bin/env python3
"""Crossword grid extraction CLI tool.

This module provides a command-line interface for extracting, straightening,
and digitizing crossword puzzle grids from images into binary matrices.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
from loguru import logger

# Import core extraction functions from library
from extract import (
    GridExtractionError,
    DimensionDetectionError,
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
    save_matrix_to_csv,
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


def cmd_extract(args: argparse.Namespace) -> None:
    """Execute the extract command.

    Args:
        args: Parsed command-line arguments
    """
    image = load_image(args.input)

    # Extract and straighten grid
    warped, max_width, max_height = extract_grid(image)

    # Save the straightened grid image
    cv2.imwrite(str(args.output), warped)
    logger.info(f"Saved straightened grid to: {args.output}")
    logger.info(f"Grid dimensions: {max_width}x{max_height} pixels")

    # Optionally create visualization
    if args.visualize:
        vis_path = args.output.parent / f"{args.output.stem}_visualization{args.output.suffix}"
        _create_visualization(image, vis_path)
        logger.info(f"Saved visualization to: {vis_path}")


def cmd_size(args: argparse.Namespace) -> None:
    """Execute the size command.

    Args:
        args: Parsed command-line arguments
    """
    image = load_image(args.input)

    # Extract and straighten the grid first
    warped, max_width, max_height = extract_grid(image)

    # Detect dimensions on the warped/straightened grid
    cell_aspect_ratio = getattr(args, 'cell_aspect_ratio', 1.0)
    cols, rows = detect_grid_dimensions(warped, expected_cell_aspect_ratio=cell_aspect_ratio)
    logger.info(f"Detected grid dimensions: {cols} columns x {rows} rows")


def cmd_convert(args: argparse.Namespace) -> None:
    """Execute the convert command.

    Args:
        args: Parsed command-line arguments
    """
    image = load_image(args.input)

    # Step 1: Extract and straighten grid
    logger.info("Step 1/3: Extracting grid...")
    warped, max_width, max_height = extract_grid(image)

    if args.visualize:
        cv2.imwrite("1_extracted_grid.jpg", warped)
        logger.debug("Saved: 1_extracted_grid.jpg")

    # Step 2: Detect dimensions
    logger.info("Step 2/3: Detecting grid dimensions...")
    cols, rows = detect_grid_dimensions(warped, expected_cell_aspect_ratio=args.cell_aspect_ratio)
    logger.info(f"Detected: {cols} columns x {rows} rows")

    # Step 3: Convert to matrix
    step_msg = "Step 3/3: Converting to matrix"
    if args.detect_dots:
        step_msg += " (with dot detection)..."
    else:
        step_msg += "..."
    logger.info(step_msg)

    grid_matrix = convert_to_matrix(
        warped,
        max_width,
        max_height,
        rows,
        cols,
        intensity_threshold=args.threshold,
        detect_dots=args.detect_dots,
        use_curved_lines=args.use_curved_lines,
        curve_smoothing=args.curve_smoothing,
    )

    # Save to CSV
    save_matrix_to_csv(grid_matrix, args.output)

    # Print matrix to console
    if args.detect_dots:
        logger.info("\nGrid Matrix (0=Black, 1=White, 2=White+Dot):")
    else:
        logger.info("\nGrid Matrix (0=Black, 1=White):")
    print(grid_matrix)

    logger.info(f"\n✓ Conversion complete! Output saved to: {args.output}")


def _create_visualization(image: cv2.Mat, output_path: Path) -> None:
    """Create visualization showing detected grid contour.

    Args:
        image: Input image
        output_path: Where to save visualization
    """
    vis_img = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
            cv2.drawContours(vis_img, [approx], -1, (0, 255, 0), 3)
            break

    cv2.imwrite(str(output_path), vis_img)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract crossword grids from images using computer vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract and straighten the grid
  %(prog)s --input crossword.jpg extract --output grid.jpg

  # Detect grid dimensions only
  %(prog)s --input crossword.jpg size

  # Full conversion to CSV
  %(prog)s --input crossword.jpg convert --output grid.csv

  # Use custom intensity threshold
  %(prog)s --input crossword.jpg convert --output grid.csv --threshold 150

  # Adjust curve smoothing for warped grids
  %(prog)s --input crossword.jpg convert --curve-smoothing 200

  # Disable curved line detection (use straight lines only)
  %(prog)s --input crossword.jpg convert --no-curved-lines

  # Enable verbose logging
  %(prog)s --input crossword.jpg --verbose convert --output grid.csv
        """,
    )

    # Global arguments
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to input crossword image (JPEG, PNG, etc.)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (TRACE level) logging",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract and straighten the crossword grid",
    )
    extract_parser.add_argument(
        "-o", "--output",
        type=Path,
        default="extracted_grid.jpg",
        help="Path for output image file (default: extracted_grid.jpg)",
    )
    extract_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization showing detected contour",
    )

    # Size command
    size_parser = subparsers.add_parser(
        "size",
        help="Detect and display grid dimensions (columns x rows)",
    )
    size_parser.add_argument(
        "--cell-aspect-ratio",
        type=float,
        default=1.0,
        help="Expected cell width/height ratio (default: 1.0 for square cells, >1.0 for wider, <1.0 for taller)",
    )

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Full pipeline: extract, detect dimensions, and convert to CSV",
    )
    convert_parser.add_argument(
        "-o", "--output",
        type=Path,
        default="crossword_grid.csv",
        help="Path for output CSV file (default: crossword_grid.csv)",
    )
    convert_parser.add_argument(
        "-t", "--threshold",
        type=int,
        help="Manual intensity threshold for black/white classification (auto-detected if not specified)",
    )
    convert_parser.add_argument(
        "--detect-dots",
        action="store_true",
        default=True,
        help="Detect black dots in white cells marking solution letters (default: enabled)",
    )
    convert_parser.add_argument(
        "--no-detect-dots",
        action="store_false",
        dest="detect_dots",
        help="Disable dot detection (output only 0=black, 1=white)",
    )
    convert_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save intermediate visualization images",
    )
    convert_parser.add_argument(
        "--use-curved-lines",
        action="store_true",
        default=True,
        help="Use curved line detection for adaptive cell extraction (default: enabled)",
    )
    convert_parser.add_argument(
        "--no-curved-lines",
        action="store_false",
        dest="use_curved_lines",
        help="Disable curved detection, use straight lines only",
    )
    convert_parser.add_argument(
        "--curve-smoothing",
        type=float,
        default=100.0,
        help="Smoothing factor for curved line detection (10-500, default: 100)",
    )
    convert_parser.add_argument(
        "--cell-aspect-ratio",
        type=float,
        default=1.0,
        help="Expected cell width/height ratio (default: 1.0 for square cells, >1.0 for wider, <1.0 for taller)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "INFO")

    try:
        # Execute command
        if args.command == "extract":
            cmd_extract(args)
        elif args.command == "size":
            cmd_size(args)
        elif args.command == "convert":
            cmd_convert(args)

    except (FileNotFoundError, ValueError, GridExtractionError, DimensionDetectionError) as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
