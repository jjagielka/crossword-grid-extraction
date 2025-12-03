"""Core image processing library for crossword grid extraction.

This module provides the fundamental computer vision functions for:
- Extracting and straightening crossword grids from images
- Detecting grid dimensions using projection profile analysis
- Converting grids to binary matrices (black/white cell representation)

These functions are designed to be used by both CLI and MCP server interfaces.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks
from loguru import logger


class GridExtractionError(Exception):
    """Raised when grid extraction fails."""

    pass


class DimensionDetectionError(Exception):
    """Raised when grid dimension detection fails."""

    pass


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort corner points in canonical order for perspective transform.

    Args:
        pts: Array of 4 corner points with shape (4, 2)

    Returns:
        Array of 4 points ordered as: top-left, top-right, bottom-right, bottom-left

    Note:
        Uses sum and difference of coordinates to identify corners:
        - Smallest sum = top-left
        - Largest sum = bottom-right
        - Smallest diff = top-right
        - Largest diff = bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    point_sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(point_sum)]  # top-left
    rect[2] = pts[np.argmax(point_sum)]  # bottom-right
    point_diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(point_diff)]  # top-right
    rect[3] = pts[np.argmax(point_diff)]  # bottom-left
    return rect


def extract_grid(
    img: np.ndarray, contour_epsilon: float = 0.02, adaptive_block_size: int = 11, adaptive_c: int = 2
) -> tuple[np.ndarray, int, int]:
    """Extract and straighten crossword grid from image using perspective transform.

    Args:
        img: Input image as BGR numpy array
        contour_epsilon: Approximation accuracy for contour detection (0.01-0.05 typical)
        adaptive_block_size: Block size for adaptive thresholding (must be odd)
        adaptive_c: Constant subtracted from weighted mean in adaptive threshold

    Returns:
        Tuple of (warped_image, width, height) where:
            - warped_image: Straightened grid image
            - width: Width of straightened image in pixels
            - height: Height of straightened image in pixels

    Raises:
        GridExtractionError: If no quadrilateral grid contour is detected

    Note:
        Requires the crossword grid to be the largest quadrilateral in the image.
    """
    if img is None:
        raise GridExtractionError("Input image is None")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Preprocessing for Grid Detection
    # Use adaptive threshold to handle shadows and lighting variations
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c
    )

    # Find the largest quadrilateral contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    display_cnt = None
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, contour_epsilon * perimeter, True)
        if len(approx) == 4:
            display_cnt = approx
            break

    if display_cnt is None:
        raise GridExtractionError(
            "Could not detect the grid contour. Ensure the crossword is the "
            "largest quadrilateral object in the image."
        )

    logger.debug(f"Found grid contour with area: {cv2.contourArea(display_cnt)}")

    # 3. Perspective Transform (Straighten)
    pts = display_cnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute new dimensions based on corner distances
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32"
    )

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, transform_matrix, (max_width, max_height))

    logger.debug(f"Extracted grid with dimensions: {max_width}x{max_height}")

    return warped, max_width, max_height


def detect_grid_dimensions(
    img: np.ndarray, min_distance_factor: int = 50, prominence_factor: int = 10
) -> tuple[int, int]:
    """Detect grid dimensions using projection profile analysis.

    Args:
        img: Input image (BGR or grayscale)
        min_distance_factor: Divisor for minimum peak distance (image_size / factor)
        prominence_factor: Multiplier for peak prominence threshold (image_size * factor)

    Returns:
        Tuple of (columns, rows) detected in the grid

    Raises:
        DimensionDetectionError: If unable to detect grid dimensions
        ValueError: If image is None or invalid

    Note:
        Uses projection profiles and peak detection to find grid lines.
        Calculates median cell size for robustness against missing lines.
        Returns dimensions in (cols, rows) order to match standard convention.
    """
    if img is None:
        raise ValueError("Input image is None")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    height, width = gray.shape

    # 2. Calculate pixel projections (sum of pixel values)
    # Summing along rows gives the vertical profile (detects horizontal lines)
    # Summing along cols gives the horizontal profile (detects vertical lines)
    row_projection = np.sum(gray, axis=1)
    col_projection = np.sum(gray, axis=0)

    # 3. Analyze the profiles to find grid lines
    # We look for "peaks" in the negative projection (dark lines = dips in brightness)
    # Inverting the projection makes dark lines appear as peaks
    neg_row_proj = -row_projection
    neg_col_proj = -col_projection

    # Find peaks (grid lines) with minimum distance and prominence constraints
    # Start with a conservative min_distance to avoid missing lines
    min_row_distance = max(1, height // min_distance_factor)
    min_col_distance = max(1, width // min_distance_factor)
    row_prominence = width * prominence_factor
    col_prominence = height * prominence_factor

    peaks_rows, _ = find_peaks(neg_row_proj, distance=min_row_distance, prominence=row_prominence)
    peaks_cols, _ = find_peaks(neg_col_proj, distance=min_col_distance, prominence=col_prominence)

    # 4. Calculate dimensions using median cell size
    # This method is robust to missing or faint grid lines
    median_row_height = 0
    median_col_width = 0
    estimated_rows = 0
    estimated_cols = 0

    if len(peaks_rows) > 1:
        row_diffs = np.diff(peaks_rows)
        median_row_height = np.median(row_diffs)
        estimated_rows = round(height / median_row_height)

    if len(peaks_cols) > 1:
        col_diffs = np.diff(peaks_cols)
        median_col_width = np.median(col_diffs)
        estimated_cols = round(width / median_col_width)

        # Adaptive refinement: Detect if we have over-detection (too many peaks)
        # This happens when grid lines have thick borders or internal structure is detected
        # Check if detected columns is unreasonably high (>30 is suspicious for typical crosswords)
        if estimated_cols > 30 or len(col_diffs) > 30:
            # Try clustering: if we have many closely-spaced peaks, they might be duplicates
            # Look for natural breaks in the spacing distribution
            sorted_diffs = np.sort(col_diffs)
            spacing_gaps = np.diff(sorted_diffs)

            if len(spacing_gaps) > 0:
                max_gap_idx = np.argmax(spacing_gaps)
                max_gap = spacing_gaps[max_gap_idx]

                # If the largest gap is significant (>15% of median), try refinement
                # Lower threshold for suspected over-detection cases
                if max_gap > median_col_width * 0.15:
                    larger_spacings = sorted_diffs[max_gap_idx + 1:]
                    if len(larger_spacings) >= 3:
                        refined_min_distance = int(np.median(larger_spacings) * 0.85)
                        peaks_cols_refined, _ = find_peaks(neg_col_proj, distance=refined_min_distance, prominence=col_prominence)
                        if len(peaks_cols_refined) > 1:
                            col_diffs_refined = np.diff(peaks_cols_refined)
                            median_col_width_refined = np.median(col_diffs_refined)
                            estimated_cols_refined = round(width / median_col_width_refined)
                            if 5 <= estimated_cols_refined <= 30:
                                estimated_cols = estimated_cols_refined
                                peaks_cols = peaks_cols_refined
                                median_col_width = median_col_width_refined
                                logger.debug(f"Refined column detection (over-detection): {len(peaks_cols)} peaks → {estimated_cols} columns")

    if len(peaks_rows) > 1:
        row_diffs = np.diff(peaks_rows)
        median_row_height = np.median(row_diffs)
        estimated_rows = round(height / median_row_height)

        # Similar bimodal distribution detection for rows
        if len(row_diffs) > 4:
            sorted_diffs = np.sort(row_diffs)
            spacing_gaps = np.diff(sorted_diffs)
            if len(spacing_gaps) > 0:
                max_gap_idx = np.argmax(spacing_gaps)
                max_gap = spacing_gaps[max_gap_idx]
                if max_gap > median_row_height * 0.2:
                    larger_spacings = sorted_diffs[max_gap_idx + 1:]
                    if len(larger_spacings) >= 3:
                        refined_min_distance = int(np.median(larger_spacings) * 0.85)
                        peaks_rows_refined, _ = find_peaks(neg_row_proj, distance=refined_min_distance, prominence=row_prominence)
                        if len(peaks_rows_refined) > 1:
                            row_diffs_refined = np.diff(peaks_rows_refined)
                            median_row_height_refined = np.median(row_diffs_refined)
                            estimated_rows_refined = round(height / median_row_height_refined)
                            if 5 <= estimated_rows_refined <= 30:
                                estimated_rows = estimated_rows_refined
                                peaks_rows = peaks_rows_refined
                                median_row_height = median_row_height_refined
                                logger.debug(f"Refined row detection (bimodal distribution): {len(peaks_rows)} peaks → {estimated_rows} rows")

    # Cross-validation: Use successfully detected dimension to validate the other
    # If one dimension (usually rows) was successfully refined and the other wasn't,
    # use the cell size from the good dimension to re-detect the problematic one
    if estimated_rows > 0 and median_row_height > 0 and estimated_cols > 30:
        # Row detection succeeded but column detection seems to have too many peaks
        # Assume roughly square cells and use row cell height as minimum distance for columns
        logger.debug(f"Column detection found {estimated_cols} cols (suspicious). Using row cell size for guidance...")

        min_col_distance_cross = int(median_row_height * 0.85)
        peaks_cols_cross, _ = find_peaks(neg_col_proj, distance=min_col_distance_cross, prominence=col_prominence)

        if len(peaks_cols_cross) > 1:
            col_diffs_cross = np.diff(peaks_cols_cross)
            median_col_width_cross = np.median(col_diffs_cross)
            estimated_cols_cross = round(width / median_col_width_cross)

            if 5 <= estimated_cols_cross <= 30:
                estimated_cols = estimated_cols_cross
                peaks_cols = peaks_cols_cross
                median_col_width = median_col_width_cross
                logger.debug(f"Cross-validated column detection: {len(peaks_cols)} peaks → {estimated_cols} columns")

    logger.debug("--- Grid Dimension Analysis ---")
    logger.debug(f"Image Size: {width}x{height} pixels")
    logger.debug(f"Detected Vertical Grid Lines: {len(peaks_cols)}")
    logger.debug(f"Detected Horizontal Grid Lines: {len(peaks_rows)}")

    if median_col_width > 0 and median_row_height > 0:
        logger.debug(f"Median Cell Size: {median_col_width:.2f}px x {median_row_height:.2f}px")
        logger.debug(f"Calculated Grid Dimensions: {estimated_cols} cols x {estimated_rows} rows")

    # Validate detected dimensions
    if estimated_rows < 1 or estimated_cols < 1:
        raise DimensionDetectionError(
            f"Failed to detect valid grid dimensions. "
            f"Detected: {estimated_cols}x{estimated_rows}. "
            f"Found {len(peaks_cols)} vertical and {len(peaks_rows)} horizontal lines. "
            f"Try adjusting image contrast or ensure grid lines are visible."
        )

    # Sanity check: typical crosswords are 10-25 cells on each side
    if estimated_rows > 50 or estimated_cols > 50:
        logger.warning(
            f"Detected unusually large dimensions: {estimated_cols}x{estimated_rows}. "
            f"This may indicate detection issues."
        )

    return estimated_cols, estimated_rows


def _detect_dot_in_cell(
    cell: np.ndarray,
    dot_size_ratio: float,
    intensity_threshold: float,
) -> bool:
    """Detect if a white cell contains a black dot in the bottom-right corner.

    Args:
        cell: Grayscale cell image
        dot_size_ratio: Size of the corner region to check (0.1-0.3 of cell size)
        intensity_threshold: Threshold for black/white classification

    Returns:
        True if a black dot is detected in the bottom-right corner

    Note:
        The dot is detected by checking if the bottom-right corner region
        has significantly lower average intensity than the cell center.
    """
    cell_h, cell_w = cell.shape

    # Define the bottom-right corner region where the dot should be
    dot_region_h = int(cell_h * dot_size_ratio)
    dot_region_w = int(cell_w * dot_size_ratio)

    if dot_region_h < 3 or dot_region_w < 3:
        # Cell too small for reliable dot detection
        return False

    # Extract bottom-right corner (avoiding the very edge to skip grid lines)
    edge_margin = max(2, int(min(cell_h, cell_w) * 0.1))
    br_corner = cell[
        cell_h - dot_region_h - edge_margin : cell_h - edge_margin,
        cell_w - dot_region_w - edge_margin : cell_w - edge_margin,
    ]

    if br_corner.size == 0:
        return False

    # Calculate average intensity in the corner
    corner_intensity = np.mean(br_corner)

    # Calculate center intensity of the cell for comparison
    cell_h, cell_w = cell.shape
    margin_y = int(cell_h * 0.25)
    margin_x = int(cell_w * 0.25)
    center = cell[margin_y : cell_h - margin_y, margin_x : cell_w - margin_x]
    center_intensity = np.mean(center)

    # A dot is present if the corner is noticeably darker than the cell center
    # Use relative comparison: corner should be darker than center
    # Real dots show at least 11-12% difference in intensity
    # False positives show only ~10% difference from natural variation
    # Use 89% threshold (11% difference) as the decision boundary
    relative_threshold = center_intensity * 0.89  # Corner should be <89% of center brightness

    return corner_intensity < relative_threshold


def convert_to_matrix(
    image: np.ndarray,
    max_width: int,
    max_height: int,
    rows: int,
    cols: int,
    intensity_threshold: Optional[int] = None,
    cell_margin: float = 0.25,
    detect_dots: bool = True,
    dot_size_ratio: float = 0.20,
) -> np.ndarray:
    """Convert straightened grid image to matrix with optional dot detection.

    Args:
        image: Straightened grid image (BGR format)
        max_width: Width of the image in pixels
        max_height: Height of the image in pixels
        rows: Number of rows in the crossword grid
        cols: Number of columns in the crossword grid
        intensity_threshold: Threshold for black/white classification (auto-detected if None)
        cell_margin: Fraction of cell to crop from edges (0.0-0.5)
        detect_dots: Whether to detect black dots in white cells (solution markers)
        dot_size_ratio: Size of dot region to check in bottom-right corner (0.1-0.3)

    Returns:
        Matrix where:
        - 0 = black cell (filled)
        - 1 = white cell (empty)
        - 2 = white cell with black dot (solution letter location)

    Raises:
        ValueError: If rows or cols is less than 1

    Note:
        Uses center sampling with configurable margins to avoid grid line contamination.
        Auto-threshold uses Otsu's method if not specified.
        Black dots are detected in the bottom-right corner of white cells.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"Invalid grid dimensions: {rows}x{cols}. Must be at least 1x1.")

    # Convert to grayscale
    warped_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grid_matrix = np.zeros((rows, cols), dtype=int)

    # Calculate cell dimensions
    cell_height = max_height / rows
    cell_width = max_width / cols

    # Auto-detect threshold if not provided
    if intensity_threshold is None:
        # Use Otsu's method on the entire image to find optimal threshold
        _, binary = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        intensity_threshold = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        logger.debug(f"Auto-detected intensity threshold: {intensity_threshold:.1f}")
    else:
        logger.debug(f"Using manual intensity threshold: {intensity_threshold}")

    # Process each cell
    dot_count = 0
    for r in range(rows):
        for c in range(cols):
            y1 = int(r * cell_height)
            y2 = int((r + 1) * cell_height)
            x1 = int(c * cell_width)
            x2 = int((c + 1) * cell_width)

            # Extract cell and crop to center (to avoid grid lines)
            cell = warped_gray[y1:y2, x1:x2]
            cell_h, cell_w = cell.shape
            margin_y = int(cell_h * cell_margin)
            margin_x = int(cell_w * cell_margin)
            center = cell[margin_y : cell_h - margin_y, margin_x : cell_w - margin_x]

            avg_intensity = np.mean(center)

            # Classification: High intensity = White cell, Low intensity = Black cell
            if avg_intensity > intensity_threshold:
                # White cell - check for dot in bottom right corner
                if detect_dots:
                    has_dot = _detect_dot_in_cell(cell, dot_size_ratio, intensity_threshold)
                    grid_matrix[r, c] = 2 if has_dot else 1
                    if has_dot:
                        dot_count += 1
                else:
                    grid_matrix[r, c] = 1
            else:
                # Black cell
                grid_matrix[r, c] = 0

    # Output result
    if detect_dots:
        logger.info(f"Extracted {rows}x{cols} grid matrix (0=Black, 1=White, 2=White+Dot)")
        logger.info(
            f"Grid statistics: {np.sum(grid_matrix == 1)} white cells, "
            f"{np.sum(grid_matrix == 0)} black cells, {dot_count} cells with dots"
        )
    else:
        logger.info(f"Extracted {rows}x{cols} grid matrix (0=Black, 1=White)")
        logger.info(
            f"Grid statistics: {np.sum(grid_matrix == 1)} white cells, {np.sum(grid_matrix == 0)} black cells"
        )

    return grid_matrix


def save_matrix_to_csv(grid_matrix: np.ndarray, output_path: Path) -> None:
    """Save grid matrix to CSV file.

    Args:
        grid_matrix: Binary matrix to save
        output_path: Path where CSV should be saved
    """
    np.savetxt(output_path, grid_matrix, fmt="%d", delimiter=",")
    logger.info(f"Saved matrix to {output_path}")
