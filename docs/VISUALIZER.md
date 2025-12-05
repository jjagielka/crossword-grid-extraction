# Crossword Grid Extraction Visualizer

A GTK4-based GUI application for visualizing the detection steps and tuning extraction parameters in real-time.

## Features

- **Step-by-step visualization**:
  - Original image
  - Contour detection (grid boundary)
  - Extracted and straightened grid
  - Grid line detection
  - Cell classification (black/white)
  - Dot detection

- **Zoom controls**:
  - Zoom in/out (25% increments, 10%-500% range)
  - Fit to window (automatic sizing)
  - Reset to 100% (actual size)
  - Keyboard shortcuts (Ctrl++, Ctrl+-, Ctrl+0)
  - Live zoom level display in status bar

- **Parameter tuning**:
  - Intensity threshold (50-250)
  - Cell margin for sampling (0.10-0.49)
  - Dot size ratio (0.10-0.40)
  - Minimum cell size for dot detection (10-100px)
  - Enable/disable dot detection

- **Real-time updates**: Parameters automatically reprocess the image after a short delay

- **Detailed information panel**: Shows detected dimensions, cell statistics, and current parameters

## Installation

### System Requirements

The visualizer requires GTK4 and PyGObject. Install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gtk-4.0
```

**Fedora:**
```bash
sudo dnf install python3-gobject gtk4
```

**Arch Linux:**
```bash
sudo pacman -S python-gobject gtk4
```

**macOS:**
```bash
brew install pygobject3 gtk4
```


### Python Dependencies

```bash
# Install with GUI dependencies
uv pip install -e ".[gui]"

# Or using pip
pip install -e ".[gui]"
```

## Usage

### Launch the Application

```bash
# Using the launcher script
./visualizer.sh

# With a specific image
./visualizer.sh path/to/image.jpg

# Or directly with Python
python src/visualizer.py test_data/wciska_kig.jpg
```

### Workflow

1. **Open an image**: Click the "Open Image" button in the toolbar
2. **Process**: Click the "Process" button to run the extraction pipeline
3. **Explore steps**: Use the radio buttons on the left to switch between visualization steps
4. **Tune parameters**: Adjust the sliders to fine-tune detection
   - Changes automatically trigger reprocessing after 500ms
5. **View results**: Check the information panel for detected dimensions and statistics

## Visualization Steps

### 1. Original Image
Shows the input image as loaded.

### 2. Contour Detection
- All detected contours shown in **blue**
- Largest quadrilateral (detected grid boundary) highlighted in **green**
- Helps debug grid detection failures

### 3. Extracted Grid
- Straightened grid after perspective transformation
- Shows the result of the warping step
- Grid should be rectangular and aligned

### 4. Grid Lines Detection
- Detected column and row positions overlaid in **green**
- Helps verify correct dimension detection
- Lines should align with actual grid lines

### 5. Cell Classification
- Visual representation of the binary matrix
- **Dark gray** = Black cells (0)
- **White** = White cells (1)
- **Light green** = White cells with dots (2)
- Grid lines shown in gray

### 6. Dot Detection
- Original extracted grid with dot annotations
- **Red circles** mark detected dot positions (cell centers)
- **Yellow rectangles** show the dot detection region (bottom-right 20% of cell)

## Parameters

### Intensity Threshold (50-250)
- Controls black/white cell classification
- Lower values → more cells classified as white
- Higher values → more cells classified as black
- Default: 140 (auto-detected via Otsu's method)

### Cell Margin (0.10-0.49)
- Fraction of cell to crop from edges when sampling
- Higher values avoid grid line interference
- Lower values use more of the cell area
- Default: 0.25 (center 50% of cell)

### Dot Size Ratio (0.10-0.40)
- Size of corner region to check for dots (fraction of cell size)
- Larger values check bigger regions
- Smaller values focus on smaller dots
- Default: 0.20 (bottom-right 20% of cell)

### Min Cell Size (10-100px)
- Minimum cell size required for dot detection
- Prevents false positives from cell numbers in small cells
- Set lower for high-resolution images
- Default: 30px

### Enable Dot Detection
- Toggle dot detection on/off
- Disabling speeds up processing
- Useful for crosswords without solution markers

## Troubleshooting

### Application won't start
```
Error: Namespace Gtk not available
```
**Solution**: Install GTK4 and PyGObject system packages (see Installation section)

**GTK4 not available?** Use the GTK3 version (`visualizer.py`) if GTK4 is not installed on your system.

### Image processing fails
- Check the status bar for error messages
- Try adjusting the intensity threshold
- Ensure the crossword is the largest object in the image
- Use the Contour Detection view to verify grid boundary detection

### No dots detected
- Check if cell size is above the minimum threshold
- View the Dot Detection visualization to see detection regions
- Adjust the Dot Size Ratio or Min Cell Size parameters
- Ensure dots are actually present in the image

### Slow performance
- Disable auto-reprocessing by not changing parameters
- Use lower resolution images
- Disable dot detection if not needed

## Keyboard Shortcuts

### File Operations
- `Ctrl+O`: Open crossword image
- `Ctrl+P`: Process current image
- `Ctrl+Q`: Quit application

### Zoom Controls
- `Ctrl+F`: Fit image to window
- `Ctrl++` or `Ctrl+=`: Zoom in (increase by 25%)
- `Ctrl+-`: Zoom out (decrease by 25%)
- `Ctrl+0`: Reset to 100% zoom (actual size)

**Note:** All zoom shortcuts also work with the numeric keypad (e.g., `Ctrl+Keypad+` for zoom in)

## Known Limitations

1. **No undo/redo**: Parameter changes trigger immediate reprocessing
2. **No export**: Cannot save processed images or matrices from GUI (use CLI for this)
3. **Single image**: Can only process one image at a time
4. **No pan**: Zoomed images require scrolling (no click-and-drag pan yet)

## Future Enhancements

Potential improvements:
- Batch processing multiple images
- Side-by-side comparison of different parameter sets
- Export processed images and matrices
- Click-and-drag pan for zoomed images
- Mouse wheel zoom
- More keyboard shortcuts (open, process, quit)
- Parameter presets/profiles
- Grid overlay with cell coordinates
- Click on cell to inspect values

## Technical Details

### Architecture
- **UI**: GTK4 via UI Builder XML
- **Backend**: OpenCV + NumPy for image processing
- **Framework**: PyGObject (Python bindings for GLib/GTK)

### File Structure
```
src/
├── visualizer.py      # GTK4 application code
├── visualizer.ui      # GTK4 UI definition
visualizer.sh          # Launcher script
```

### Architecture
The visualizer uses modern GTK4 with:
- Modern GTK4 widgets (`GtkHeaderBar`, `GtkPicture`, `GtkFileDialog`)
- **GAction architecture** for keyboard shortcuts and commands
- Actions defined declaratively with `Gio.SimpleAction`
- Keyboard accelerators set via `set_accels_for_action()`
- Data-driven action setup pattern (minimal boilerplate)
- Better integration with modern Linux desktops
- Improved performance and memory usage

### GAction Architecture
Modern action-based architecture:
- All commands (Open, Process, Zoom, Quit) are `GAction` objects
- Toolbar buttons trigger actions via `action-name` property
- Keyboard shortcuts registered with `set_accels_for_action()`
- Follows GNOME Human Interface Guidelines (HIG)
- Enables future features like application menus and customizable shortcuts

### Performance
- Image loading: Instant
- Processing time: 80-200ms per image (typical)
- Auto-reprocess delay: 500ms (debounced)

## Related Documentation

- **README.md** - Main project documentation
- **docs/DOT_DETECTION_FEATURE.md** - Dot detection algorithm details
- **docs/LIBRARY_USAGE.md** - Using the core library programmatically
