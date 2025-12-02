"""Integration and unit tests for crossword extraction."""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add src directory to path to import crossword module
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crossword import (
    order_points,
    extract_grid,
    convert,
    detect_grid_dimensions,
    Application,
    GridExtractionError,
    DimensionDetectionError,
)


class TestIntegration:
    """Integration tests for the full conversion pipeline."""

    def test_full_pipeline_crossword1(
        self, test_image_crossword1, expected_result1, temp_output_dir
    ):
        """Test full pipeline on nyt1.png matches expected result."""
        # Load image
        img = cv2.imread(str(test_image_crossword1))
        assert img is not None, "Failed to load nyt1.png"

        # Extract grid
        warped, max_width, max_height = extract_grid(img)
        assert warped is not None
        assert max_width > 0
        assert max_height > 0

        # Detect dimensions
        cols, rows = detect_grid_dimensions(warped)
        assert cols > 0
        assert rows > 0

        # Convert to binary matrix
        output_path = temp_output_dir / "test1.csv"
        grid_matrix = convert(
            warped, max_width, max_height, rows, cols, output_path=output_path
        )

        # Verify output
        expected = np.array(expected_result1)
        assert grid_matrix.shape == expected.shape, (
            f"Shape mismatch: got {grid_matrix.shape}, expected {expected.shape}"
        )
        np.testing.assert_array_equal(
            grid_matrix, expected, err_msg="Grid matrix does not match expected result1"
        )

        # Verify CSV was created
        assert output_path.exists(), "Output CSV file was not created"

    def test_full_pipeline_crossword2(
        self, test_image_crossword2, expected_result2, temp_output_dir
    ):
        """Test full pipeline on nyt2.jpg matches expected result."""
        # Load image
        img = cv2.imread(str(test_image_crossword2))
        assert img is not None, "Failed to load nyt2.jpg"

        # Extract grid
        warped, max_width, max_height = extract_grid(img)
        assert warped is not None
        assert max_width > 0
        assert max_height > 0

        # Detect dimensions
        cols, rows = detect_grid_dimensions(warped)
        assert cols > 0
        assert rows > 0

        # Convert to binary matrix
        output_path = temp_output_dir / "test2.csv"
        grid_matrix = convert(
            warped, max_width, max_height, rows, cols, output_path=output_path
        )

        # Verify output
        expected = np.array(expected_result2)
        assert grid_matrix.shape == expected.shape, (
            f"Shape mismatch: got {grid_matrix.shape}, expected {expected.shape}"
        )
        np.testing.assert_array_equal(
            grid_matrix, expected, err_msg="Grid matrix does not match expected result2"
        )

        # Verify CSV was created
        assert output_path.exists(), "Output CSV file was not created"

    def test_application_convert_crossword1(
        self, test_image_crossword1, expected_result1, temp_output_dir
    ):
        """Test Application.convert() on nyt1.png."""
        output_path = temp_output_dir / "app_test1.csv"

        app = Application(input=str(test_image_crossword1), verbose=False)
        app.convert(output=str(output_path))

        # Read the generated CSV
        generated = np.loadtxt(output_path, delimiter=",", dtype=int)
        expected = np.array(expected_result1)

        np.testing.assert_array_equal(
            generated, expected, err_msg="Application output does not match expected"
        )

    def test_application_convert_crossword2(
        self, test_image_crossword2, expected_result2, temp_output_dir
    ):
        """Test Application.convert() on nyt2.jpg."""
        output_path = temp_output_dir / "app_test2.csv"

        app = Application(input=str(test_image_crossword2), verbose=False)
        app.convert(output=str(output_path))

        # Read the generated CSV
        generated = np.loadtxt(output_path, delimiter=",", dtype=int)
        expected = np.array(expected_result2)

        np.testing.assert_array_equal(
            generated, expected, err_msg="Application output does not match expected"
        )


class TestOrderPoints:
    """Unit tests for order_points function."""

    def test_order_points_basic(self):
        """Test order_points with a simple rectangle."""
        # Create points in random order
        pts = np.array([[100, 100], [400, 100], [400, 300], [100, 300]], dtype="float32")

        ordered = order_points(pts)

        # Check that we got 4 points back
        assert ordered.shape == (4, 2)

        # Top-left should have smallest sum
        assert np.allclose(ordered[0], [100, 100])
        # Bottom-right should have largest sum
        assert np.allclose(ordered[2], [400, 300])

    def test_order_points_scrambled(self):
        """Test order_points with scrambled input."""
        # Points in non-standard order
        pts = np.array([[400, 300], [100, 100], [400, 100], [100, 300]], dtype="float32")

        ordered = order_points(pts)

        # Verify canonical order: TL, TR, BR, BL
        assert ordered[0][0] < ordered[1][0]  # TL.x < TR.x
        assert ordered[0][1] < ordered[2][1]  # TL.y < BR.y
        assert ordered[2][0] > ordered[3][0]  # BR.x > BL.x


class TestExtractGrid:
    """Unit tests for extract_grid function."""

    def test_extract_grid_returns_tuple(self, test_image_crossword1):
        """Test that extract_grid returns a 3-tuple."""
        img = cv2.imread(str(test_image_crossword1))
        result = extract_grid(img)

        assert isinstance(result, tuple)
        assert len(result) == 3

        warped, max_width, max_height = result
        assert isinstance(warped, np.ndarray)
        assert isinstance(max_width, int)
        assert isinstance(max_height, int)

    def test_extract_grid_dimensions_positive(self, test_image_crossword1):
        """Test that extracted grid has positive dimensions."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)

        assert max_width > 0
        assert max_height > 0
        assert warped.shape[0] == max_height
        assert warped.shape[1] == max_width

    def test_extract_grid_none_image(self):
        """Test that extract_grid raises error on None image."""
        with pytest.raises(GridExtractionError, match="Input image is None"):
            extract_grid(None)

    def test_extract_grid_configurable_epsilon(self, test_image_crossword1):
        """Test that contour_epsilon parameter works."""
        img = cv2.imread(str(test_image_crossword1))

        # Should work with different epsilon values
        result1 = extract_grid(img, contour_epsilon=0.01)
        result2 = extract_grid(img, contour_epsilon=0.03)

        assert result1 is not None
        assert result2 is not None


class TestDetectGridDimensions:
    """Unit tests for detect_grid_dimensions function."""

    def test_detect_dimensions_returns_tuple(self, test_image_crossword1):
        """Test that detect_grid_dimensions returns (cols, rows) tuple."""
        img = cv2.imread(str(test_image_crossword1))
        warped, _, _ = extract_grid(img)

        result = detect_grid_dimensions(warped)

        assert isinstance(result, tuple)
        assert len(result) == 2

        cols, rows = result
        assert isinstance(cols, (int, np.integer))
        assert isinstance(rows, (int, np.integer))

    def test_detect_dimensions_positive(self, test_image_crossword1):
        """Test that detected dimensions are positive."""
        img = cv2.imread(str(test_image_crossword1))
        warped, _, _ = extract_grid(img)

        cols, rows = detect_grid_dimensions(warped)

        assert cols > 0
        assert rows > 0

    def test_detect_dimensions_reasonable_range(self, test_image_crossword1):
        """Test that detected dimensions are in reasonable range."""
        img = cv2.imread(str(test_image_crossword1))
        warped, _, _ = extract_grid(img)

        cols, rows = detect_grid_dimensions(warped)

        # Typical crosswords are 5-30 cells
        assert 5 <= cols <= 30
        assert 5 <= rows <= 30

    def test_detect_dimensions_none_image(self):
        """Test that detect_grid_dimensions raises error on None image."""
        with pytest.raises(ValueError, match="Input image is None"):
            detect_grid_dimensions(None)


class TestConvert:
    """Unit tests for convert function."""

    def test_convert_returns_matrix(self, test_image_crossword1):
        """Test that convert returns a numpy array."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        result = convert(warped, max_width, max_height, rows, cols)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int_

    def test_convert_correct_shape(self, test_image_crossword1):
        """Test that convert returns correct shape."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        result = convert(warped, max_width, max_height, rows, cols)

        assert result.shape == (rows, cols)

    def test_convert_binary_values(self, test_image_crossword1):
        """Test that convert returns only 0s and 1s."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        result = convert(warped, max_width, max_height, rows, cols)

        # Should only contain 0 and 1
        assert np.all((result == 0) | (result == 1))

    def test_convert_invalid_dimensions(self, test_image_crossword1):
        """Test that convert raises error on invalid dimensions."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)

        with pytest.raises(ValueError, match="Invalid grid dimensions"):
            convert(warped, max_width, max_height, 0, 10)

        with pytest.raises(ValueError, match="Invalid grid dimensions"):
            convert(warped, max_width, max_height, 10, -1)

    def test_convert_saves_to_file(self, test_image_crossword1, temp_output_dir):
        """Test that convert saves CSV file when output_path provided."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        output_path = temp_output_dir / "test_convert.csv"
        result = convert(
            warped, max_width, max_height, rows, cols, output_path=output_path
        )

        assert output_path.exists()

        # Verify saved file matches returned array
        loaded = np.loadtxt(output_path, delimiter=",", dtype=int)
        np.testing.assert_array_equal(loaded, result)

    def test_convert_manual_threshold(self, test_image_crossword1):
        """Test that manual threshold parameter works."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        # Convert with auto-threshold
        result_auto = convert(warped, max_width, max_height, rows, cols)

        # Convert with manual threshold
        result_manual = convert(
            warped, max_width, max_height, rows, cols, intensity_threshold=128
        )

        # Both should be valid binary matrices
        assert np.all((result_auto == 0) | (result_auto == 1))
        assert np.all((result_manual == 0) | (result_manual == 1))


class TestApplication:
    """Unit tests for Application class."""

    def test_application_init_valid_image(self, test_image_crossword1):
        """Test Application initialization with valid image."""
        app = Application(input=str(test_image_crossword1), verbose=False)

        assert app.image_path == Path(test_image_crossword1)
        assert app.image is not None
        assert isinstance(app.image, np.ndarray)

    def test_application_init_nonexistent_file(self, temp_output_dir):
        """Test Application raises error for nonexistent file."""
        fake_path = temp_output_dir / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            Application(input=str(fake_path))

    def test_application_init_invalid_image(self, temp_output_dir):
        """Test Application raises error for invalid image file."""
        # Create a text file, not an image
        fake_image = temp_output_dir / "fake.jpg"
        fake_image.write_text("This is not an image")

        with pytest.raises(ValueError, match="Failed to load image"):
            Application(input=str(fake_image))

    def test_application_extract(self, test_image_crossword1, temp_output_dir):
        """Test Application.extract() method."""
        output_path = temp_output_dir / "extracted.jpg"

        app = Application(input=str(test_image_crossword1), verbose=False)
        app.extract(output=str(output_path))

        assert output_path.exists()

        # Verify it's a valid image
        extracted = cv2.imread(str(output_path))
        assert extracted is not None

    def test_application_extract_with_visualize(
        self, test_image_crossword1, temp_output_dir
    ):
        """Test Application.extract() with visualization."""
        output_path = temp_output_dir / "extracted_vis.jpg"

        app = Application(input=str(test_image_crossword1), verbose=False)
        app.extract(output=str(output_path), visualize=True)

        assert output_path.exists()

        # Visualization file should also exist
        vis_path = output_path.parent / f"{output_path.stem}_visualization{output_path.suffix}"
        assert vis_path.exists()

    def test_application_size(self, test_image_crossword1):
        """Test Application.size() method."""
        app = Application(input=str(test_image_crossword1), verbose=False)

        # Should not raise an exception
        app.size()


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_no_contour_detected(self):
        """Test behavior when no grid contour is detected."""
        # Create a blank image with no grid
        blank = np.ones((500, 500, 3), dtype=np.uint8) * 255

        with pytest.raises(GridExtractionError, match="Could not detect"):
            extract_grid(blank)

    def test_dimension_detection_failure(self):
        """Test behavior when dimension detection fails."""
        # Create a uniform white image (no grid lines to detect)
        uniform_img = np.ones((200, 200), dtype=np.uint8) * 255

        # Dimension detection should fail (no peaks detected)
        with pytest.raises(DimensionDetectionError, match="Failed to detect"):
            detect_grid_dimensions(uniform_img)


class TestOutputFormat:
    """Tests for output format consistency."""

    def test_csv_output_format(self, test_image_crossword1, temp_output_dir):
        """Test that CSV output has correct format."""
        output_path = temp_output_dir / "format_test.csv"

        app = Application(input=str(test_image_crossword1), verbose=False)
        app.convert(output=str(output_path))

        # Read CSV manually
        with open(output_path, "r") as f:
            lines = f.readlines()

        # Each line should be comma-separated integers
        for line in lines:
            line = line.strip()
            if line:
                values = line.split(",")
                for val in values:
                    assert val in ["0", "1"], f"Invalid value in CSV: {val}"

    def test_matrix_representation_matches_csv(
        self, test_image_crossword1, temp_output_dir
    ):
        """Test that returned matrix matches saved CSV."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        output_path = temp_output_dir / "matrix_csv_test.csv"
        matrix = convert(warped, max_width, max_height, rows, cols, output_path=output_path)

        # Load CSV
        csv_matrix = np.loadtxt(output_path, delimiter=",", dtype=int)

        # Should be identical
        np.testing.assert_array_equal(matrix, csv_matrix)


class TestConfigurability:
    """Tests for configurable parameters."""

    def test_different_cell_margins(self, test_image_crossword1):
        """Test that cell_margin parameter affects results."""
        img = cv2.imread(str(test_image_crossword1))
        warped, max_width, max_height = extract_grid(img)
        cols, rows = detect_grid_dimensions(warped)

        # Try different margins
        result1 = convert(warped, max_width, max_height, rows, cols, cell_margin=0.1)
        result2 = convert(warped, max_width, max_height, rows, cols, cell_margin=0.4)

        # Both should be valid
        assert result1.shape == (rows, cols)
        assert result2.shape == (rows, cols)

    def test_different_detection_factors(self, test_image_crossword1):
        """Test that detection factors can be adjusted."""
        img = cv2.imread(str(test_image_crossword1))
        warped, _, _ = extract_grid(img)

        # Try different detection parameters
        cols1, rows1 = detect_grid_dimensions(warped, min_distance_factor=40)
        cols2, rows2 = detect_grid_dimensions(warped, min_distance_factor=60)

        # Both should return valid dimensions
        assert cols1 > 0 and rows1 > 0
        assert cols2 > 0 and rows2 > 0
