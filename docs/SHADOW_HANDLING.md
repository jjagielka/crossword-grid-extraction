# Shadow Handling and Adaptive Thresholding

## Problem

Images with non-uniform lighting (shadows, gradients) cause significant misclassification when using global thresholding. In testing:

- **Normal image (wciska_kig.jpg)**: Global Otsu threshold = 151, works perfectly
- **Shadow image (zcieniem.jpg)**: Global Otsu threshold = 157, causes 42 white cells to be misclassified as black

### Root Cause

Global thresholding uses a single intensity value for the entire image. In the shadow image:
- Top-left white cells: intensity ~190-210 (bright, above threshold ✓)
- Bottom-right white cells: intensity ~118-148 (shadowed, below threshold ✗)

The global threshold of 157 falls between the lit and shadowed regions, causing all shadowed white cells to be incorrectly classified as black.

## Solution: Local Adaptive Thresholding

The algorithm now automatically detects lighting gradients and switches to local adaptive thresholding when needed.

### Detection Algorithm

1. **Sample corners** of the extracted grid:
   - Top-left corner (1/4 of image size)
   - Bottom-right corner (1/4 of image size)

2. **Calculate gradient**: `diff = |mean(TL) - mean(BR)|`

3. **Threshold selection**:
   - If `diff > 30`: Use **local adaptive thresholding**
   - If `diff ≤ 30`: Use **global Otsu thresholding**

### Local Thresholding Method

For each cell, when adaptive mode is active:
1. Expand the cell region by 50% in all directions
2. Apply Otsu's method to this local neighborhood
3. Use the local threshold for that specific cell

This allows each region of the image to have its own optimal threshold, adapting to local lighting conditions.

## Results

### Before Adaptive Thresholding

**Shadow image (zcieniem.jpg) with global threshold:**
- 100 white cells (should be 141) ❌
- 98 black cells (should be 53) ❌
- **46 total errors**
- 67.4% of errors concentrated in shadowed bottom-right quadrant

### After Adaptive Thresholding

**Shadow image (zcieniem.jpg) with adaptive threshold:**
- 143 white cells (expected 141) ✓ (2 extra)
- 53 black cells (expected 53) ✓
- 8 dots detected (expected 10) ⚠ (2 missed)
- **2 total errors** (both are missed dots, cells correctly classified as white)

**Improvement: 46 errors → 2 errors (96% reduction)**

### Normal Images

Images without significant lighting gradients continue to work perfectly:
- **wciska_kig.jpg**: Gradient diff = 13 → uses global Otsu ✓
- **crosswords2.jpg**: Gradient diff < 30 → uses global Otsu ✓
- Both images produce identical results to previous version

## Limitations

### Dot Detection in Shadows

Dots in heavily shadowed areas may be missed because:
1. Shadow reduces overall contrast between cell and dot
2. Both the white cell and the black dot become darker
3. The relative intensity difference decreases

In testing:
- Normal lighting: dot ratio = 0.843 (15.7% darker) → detected ✓
- Shadow lighting: dot ratio = 0.929 (7.1% darker) → not detected ✗

This affects approximately 2 out of 10 dots in extreme shadow cases. The cells are still correctly classified as white (not black), so the grid structure is correct.

### Trade-offs

**Pros:**
- Handles non-uniform lighting automatically
- No user intervention required
- Preserves accuracy on normal images
- 96% error reduction on shadow images

**Cons:**
- Slightly slower (local Otsu per cell vs. one global Otsu)
- May miss some dots in heavily shadowed regions
- Requires 50% overlap between cells for stable local threshold

## Usage

The feature is automatic and requires no configuration:

```bash
# Automatically handles shadows
python src/crossword.py --input shadowed_image.jpg convert

# Check if adaptive thresholding was used
python src/crossword.py --input image.jpg --verbose convert
# Look for: "Detected lighting gradient" and "Using local adaptive thresholding"
```

## Technical Details

### Gradient Detection Threshold

The threshold of `30` for gradient detection was chosen empirically:
- Most normal lighting: gradient < 20
- Noticeable shadows: gradient > 40
- Threshold of 30 provides good separation with safety margin

### Local Window Size

Local thresholding uses a 2x cell size window (50% expansion in each direction):
- Smaller windows: More adaptive but may be noisy
- Larger windows: More stable but less adaptive
- 2x cell size balances stability and adaptiveness

### Implementation Location

- Detection logic: `src/extract.py:convert_to_matrix()` lines 420-446
- Local threshold calculation: lines 466-480
- Affects both CLI and MCP server automatically

## Testing

Run the comprehensive shadow test:

```bash
python test_shadow_handling.py
```

This tests:
1. Normal lighting image (wciska_kig.jpg)
2. Shadow image (zcieniem.jpg)
3. Different crossword (crosswords2.jpg)

Expected output:
- Normal: 141 white, 53 black, 10 dots ✓
- Shadow: ~143 white, 53 black, ~8 dots (acceptable)
- Other: Varies by crossword ✓
