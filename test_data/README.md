# Test Data

This folder contains test images and expected results for the test suite.

## Test Images

- **nyt1.png** - First NYT crossword test image (15×15 grid, no dots)
- **nyt2.jpg** - Second NYT crossword test image (21×21 grid, no dots, but currently detected as 20×21 due to perspective distortion from angled photo. Very small cells ~19×15px with cell numbers.)
- **wciska_kig.jpg** - Test image with normal lighting (17×12 grid, 10 dots)
- **zcieniem.jpg** - Test image with shadows (17×12 grid, 10 dots)
- **crosswords2.jpg** - Different crossword (17×12 grid, 7 dots)
- **jolka.jpg** - Portrait format crossword (11×19 grid, 6 dots)

## Expected Results

- **nyt1_result.txt** - Expected binary matrix for nyt1.png (15×15)
- **nyt2_result.txt** - Expected binary matrix for nyt2.jpg (21×21 ground truth, but may contain detection errors)

## Format

The result files contain numpy array representations with:
- `0` = Black cell (blocked)
- `1` = White cell (fillable)

Example format:
```
[[0 1 0 1 0 1 0 1 0 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 0]
 ...]
```

## Adding New Test Cases

To add new test data:

1. Add crossword image: `test_data/crosswordN.jpg`
2. Generate expected result:
   ```bash
   python crossword.py --input=test_data/crosswordN.jpg convert --output=test_data/resultN.txt
   ```
3. Manually verify the output is correct
4. Add fixtures in `tests/conftest.py`
5. Add test in `tests/test_crossword.py`

## Usage in Tests

These files are loaded by pytest fixtures defined in `tests/conftest.py`:
- `test_image_crossword1` - Loads nyt1.png
- `test_image_crossword2` - Loads nyt2.jpg
- `expected_result1` - Loads and parses nyt1_result.txt
- `expected_result2` - Loads and parses nyt2_result.txt
