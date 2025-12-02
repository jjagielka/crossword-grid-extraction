"""Pytest configuration and fixtures for crossword-read tests."""

from pathlib import Path
import pytest
import numpy as np


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_image_crossword1(project_root):
    """Return path to nyt1.png test image."""
    path = project_root / "test_data" / "nyt1.png"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path


@pytest.fixture
def test_image_crossword2(project_root):
    """Return path to nyt2.jpg test image."""
    path = project_root / "test_data" / "nyt2.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found: {path}")
    return path


@pytest.fixture
def expected_result1(project_root):
    """Return expected result for nyt1.png."""
    path = project_root / "test_data" / "nyt1_result.txt"
    if not path.exists():
        pytest.skip(f"Expected result file not found: {path}")

    # Parse the numpy array from the text file
    with open(path, 'r') as f:
        content = f.read()

    # Convert numpy string representation to actual array
    # Replace multiple spaces with commas for proper Python list syntax
    import re
    content = content.strip()
    # Replace all whitespace sequences between digits with commas
    content = re.sub(r'(\d)\s+(\d)', r'\1,\2', content)
    # Multiple passes to handle all adjacent pairs
    while re.search(r'(\d)\s+(\d)', content):
        content = re.sub(r'(\d)\s+(\d)', r'\1,\2', content)
    # Fix row separators - add commas between ] [
    content = re.sub(r'\]\s+\[', '],[', content)
    # Evaluate to get Python list and convert to numpy array
    return np.array(eval(content))


@pytest.fixture
def expected_result2(project_root):
    """Return expected result for nyt2.jpg."""
    path = project_root / "test_data" / "nyt2_result.txt"
    if not path.exists():
        pytest.skip(f"Expected result file not found: {path}")

    # Parse the numpy array from the text file
    with open(path, 'r') as f:
        content = f.read()

    # Convert numpy string representation to actual array
    import re
    content = content.strip()
    # Replace all whitespace sequences between digits with commas
    content = re.sub(r'(\d)\s+(\d)', r'\1,\2', content)
    # Multiple passes to handle all adjacent pairs
    while re.search(r'(\d)\s+(\d)', content):
        content = re.sub(r'(\d)\s+(\d)', r'\1,\2', content)
    # Fix row separators - add commas between ] [
    content = re.sub(r'\]\s+\[', '],[', content)
    # Evaluate to get Python list and convert to numpy array
    return np.array(eval(content))


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
