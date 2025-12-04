# Aspect Ratio Validation

## Concept

Since crossword cells are typically **square**, we can use aspect ratios as a validation tool to detect incorrect dimension detection.

## Key Relationships

For square-celled crosswords:

1. **Image ratio ≈ Grid ratio**:
   ```
   image_width / image_height ≈ columns / rows
   ```

2. **Cells are square**:
   ```
   cell_width ≈ cell_height
   cell_aspect_ratio ≈ 1.0
   ```

## Validation Thresholds

The implementation checks two conditions:

### 1. Image/Grid Ratio Match
```python
image_ratio = image_width / image_height
grid_ratio = cols / rows
ratio_error = abs(image_ratio - grid_ratio) / image_ratio * 100

if ratio_error > 10%:
    # Warning: possible incorrect dimension detection
```

**Typical values**:
- Correct detection: 0.8% - 2.7% error
- Wrong detection: 50% - 77% error

### 2. Cell Squareness
```python
cell_aspect = cell_width / cell_height
cell_squareness_error = abs(cell_aspect - 1.0) * 100

if cell_squareness_error > 15%:
    # Warning: cells not square or detection error
```

**Typical values**:
- Square cells: 1.4% - 2.8% error
- Wrong detection: 40% - 45% error

## Example: jolka.jpg

### Correct Detection (11×19)
- Image ratio: 0.595
- Grid ratio: 0.579
- **Ratio error: 2.7%** ✓
- Cell size: 114.3×111.2 pixels
- Cell aspect: 1.028
- **Cell error: 2.8%** ✓

### Wrong Detection (20×19, before fix)
- Image ratio: 0.595
- Grid ratio: 1.053
- **Ratio error: 76.9%** ⚠ (triggers warning)
- Cell size: 62.9×111.2 pixels
- Cell aspect: 0.565
- **Cell error: 43.5%** ⚠ (triggers warning)

## Benefits

1. **Early error detection**: Catches dimension errors before conversion
2. **No ground truth needed**: Self-validating based on physical constraints
3. **Clear indicators**: Errors jump from ~2% to 50-80% for wrong detections
4. **Helps debugging**: Pinpoints whether rows or columns are wrong

## Test Results

All test images show excellent aspect ratio validation:

| Image | Dimensions | Ratio Error | Cell Error | Status |
|-------|-----------|-------------|------------|--------|
| wciska_kig.jpg | 17×12 | 2.0% | 1.4% | ✓ Pass |
| crosswords2.jpg | 17×12 | 0.8% | 0.8% | ✓ Pass |
| zcieniem.jpg | 17×12 | 1.4% | 1.3% | ✓ Pass |
| jolka.jpg | 11×19 | 2.7% | 2.8% | ✓ Pass |

## Implementation

The validation is automatically performed during `detect_grid_dimensions()` and logs:
- DEBUG: Aspect ratio metrics for all detections
- WARNING: When errors exceed thresholds (10% for ratio, 15% for squareness)

No user configuration needed - it's a passive validation check that helps identify issues.

## Limitations

This validation assumes:
1. Crossword cells are square (typical for most crosswords)
2. Grid is uniform (all cells same size)
3. Perspective correction was successful

For non-square crosswords (rare), warnings may be false positives. However, the vast majority of crosswords use square cells, making this a valuable validation tool.

## Future Enhancements

Could potentially use aspect ratio mismatch to:
1. **Auto-correct** dimension detection by trying alternative peak detection parameters
2. **Suggest** likely correct dimensions based on aspect ratio
3. **Score** different detection hypotheses and pick best match

For now, it serves as an excellent diagnostic tool during development and debugging.
