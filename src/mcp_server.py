#!/usr/bin/env python3
"""MCP server for crossword grid extraction.

This server exposes crossword grid extraction functionality to LLMs via the Model Context Protocol (MCP).
It allows LLMs to convert crossword images into binary matrices.
"""

import sys
from typing import Optional

from fastmcp import FastMCP
from loguru import logger

from utils import (
    load_image_from_base64,
    process_crossword_image,
    format_matrix,
)

# Import core extraction functions from library for specific information retrieval
from extract import (
    extract_grid,
    detect_grid_dimensions,
    GridExtractionError,
    DimensionDetectionError,
)

# Initialize FastMCP server
mcp = FastMCP("crossword-extractor")


@mcp.tool()
def extract_crossword_grid(
    image_base64: str,
    output_format: str = "csv",
    intensity_threshold: Optional[int] = None,
    detect_dots: bool = True,
    use_curved_lines: bool = True,
    curve_smoothing: float = 100.0,
    expected_cell_aspect_ratio: float = 1.0,
) -> str:
    """Extract a crossword grid from an image and convert it to a matrix.

    This tool performs the full pipeline:
    1. Extracts and straightens the crossword grid from the image
    2. Detects the grid dimensions (rows × columns)
    3. Converts the grid to a matrix with values:
       - 0 = black cell (filled)
       - 1 = white cell (empty)
       - 2 = white cell with black dot (solution letter location)

    Args:
        image_base64: Base64-encoded image data (JPEG, PNG, etc.)
        output_format: Output format - "csv" for comma-separated values, "array" for numpy array string, or "json" for JSON array
        intensity_threshold: Optional manual threshold for black/white classification (auto-detected using Otsu's method if not provided)
        detect_dots: Whether to detect black dots in white cells marking solution letters (default: True)
        use_curved_lines: Use curved line detection for adaptive cell extraction (default: True)
        curve_smoothing: Smoothing factor for curved line detection, range 10-500 (default: 100.0)
        expected_cell_aspect_ratio: Expected width/height ratio of cells (default: 1.0 for square cells, >1.0 for wider, <1.0 for taller)

    Returns:
        String containing the grid matrix in the requested format, along with metadata about dimensions and statistics

    Raises:
        ValueError: If image data is invalid or output format is unsupported
        GridExtractionError: If the grid cannot be detected in the image
        DimensionDetectionError: If the grid dimensions cannot be determined

    Example:
        For a 17×12 crossword grid, returns CSV format like:
        ```
        Detected: 17 columns × 12 rows
        Grid statistics: 165 white cells, 39 black cells

        0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1
        1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0
        ...
        ```

    Notes:
        - The crossword must be the largest quadrilateral object in the image
        - Works best with clear, well-lit images
        - Grid lines should be reasonably visible
        - Typical crossword sizes (10-25 cells) work best
        - Auto-threshold uses Otsu's method for optimal black/white separation
    """
    try:
        # Load image from base64 using central utility
        image = load_image_from_base64(image_base64)
        logger.info(f"Loaded image with dimensions: {image.shape[1]}×{image.shape[0]} pixels")

        # Use central pipeline to process image
        grid_matrix, cols, rows = process_crossword_image(
            image,
            intensity_threshold=intensity_threshold,
            detect_dots=detect_dots,
            use_curved_lines=use_curved_lines,
            curve_smoothing=curve_smoothing,
            expected_cell_aspect_ratio=expected_cell_aspect_ratio,
        )

        # Use central formatter for output
        return format_matrix(
            grid_matrix, cols, rows, output_format=output_format, detect_dots=detect_dots
        )

    except ValueError as e:
        logger.error(str(e))
        raise e

    except GridExtractionError as e:
        error_msg = (
            f"Grid extraction failed: {e}\n\n"
            f"Troubleshooting tips:\n"
            f"- Ensure the crossword is the largest quadrilateral object in the image\n"
            f"- Try improving image contrast or cropping closer to the grid\n"
            f"- Make sure the grid edges are clearly visible"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    except DimensionDetectionError as e:
        error_msg = (
            f"Dimension detection failed: {e}\n\n"
            f"Troubleshooting tips:\n"
            f"- Grid lines may be too faint or irregular\n"
            f"- Try using a higher quality image\n"
            f"- Ensure grid lines are reasonably visible\n"
            f"- Typical crossword sizes (10-25 cells) work best"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error during grid extraction: {type(e).__name__}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)


@mcp.tool()
def get_grid_info(image_base64: str) -> str:
    """Get information about a crossword grid without performing full conversion.

    This tool extracts the grid and detects its dimensions, but doesn't convert it to a matrix.
    Useful for quickly checking if a crossword image can be processed.

    Args:
        image_base64: Base64-encoded image data (JPEG, PNG, etc.)

    Returns:
        String with grid information including detected dimensions

    Raises:
        ValueError: If image data is invalid
        GridExtractionError: If the grid cannot be detected
        DimensionDetectionError: If dimensions cannot be determined

    Example:
        Returns:
        ```
        Image size: 1200×800 pixels
        Extracted grid: 850×650 pixels
        Detected dimensions: 17 columns × 12 rows
        Estimated cell size: 50.0×54.2 pixels
        ```
    """
    try:
        # Load image using central utility
        image = load_image_from_base64(image_base64)
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

        return result

    except (GridExtractionError, DimensionDetectionError) as e:
        raise ValueError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise ValueError(f"Unexpected error: {type(e).__name__}: {e}")

    except (GridExtractionError, DimensionDetectionError) as e:
        raise ValueError(str(e))
    except Exception as e:
        raise ValueError(f"Unexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP server for crossword grid extraction")
    parser.add_argument(
        "--http", action="store_true", help="Run in HTTP/SSE mode (for systemd deployment)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging"
    )
    args = parser.parse_args()

    # Configure logging level
    # Remove default handler and add new one with appropriate level
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Run server in appropriate mode
    if args.http:
        # Run in HTTP/SSE mode for persistent daemon deployment
        # This mode is suitable for systemd services
        logger.info("Starting MCP server in HTTP/SSE mode")
        mcp.run(transport="sse")
    else:
        # Default: Run in STDIO mode for Claude Desktop and similar clients
        # This mode is launched on-demand by the client
        logger.info("Starting MCP server in STDIO mode")
        mcp.run()
