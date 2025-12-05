#!/usr/bin/env python3
"""
Crossword Grid Extraction Visualizer

A PyGTK application for visualizing the detection steps and tuning parameters.
"""

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, GLib
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Tuple

# Import extraction functions
from extract import (
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
    GridExtractionError,
    DimensionDetectionError,
)


class CrosswordVisualizer:
    """Main application class for crossword grid extraction visualization."""

    def __init__(self):
        """Initialize the visualizer application."""
        # Load the Glade file
        self.builder = Gtk.Builder()
        glade_file = Path(__file__).parent / "visualizer.glade"
        self.builder.add_from_file(str(glade_file))

        # Get main window
        self.window = self.builder.get_object("main_window")

        # Get widgets
        self.image_display = self.builder.get_object("image_display")
        self.text_info = self.builder.get_object("text_info")
        self.statusbar = self.builder.get_object("statusbar")

        # Get controls
        self.scale_threshold = self.builder.get_object("scale_threshold")
        self.scale_cell_margin = self.builder.get_object("scale_cell_margin")
        self.scale_dot_ratio = self.builder.get_object("scale_dot_ratio")
        self.scale_min_cell_size = self.builder.get_object("scale_min_cell_size")
        self.check_detect_dots = self.builder.get_object("check_detect_dots")

        # Get radio buttons
        self.radio_original = self.builder.get_object("radio_original")
        self.radio_contours = self.builder.get_object("radio_contours")
        self.radio_extracted = self.builder.get_object("radio_extracted")
        self.radio_lines = self.builder.get_object("radio_lines")
        self.radio_cells = self.builder.get_object("radio_cells")
        self.radio_dots = self.builder.get_object("radio_dots")

        # Connect signals
        self.builder.connect_signals(self)

        # Add keyboard shortcuts for zoom
        accel_group = Gtk.AccelGroup()
        self.window.add_accel_group(accel_group)

        # Define accelerator callbacks
        def open_callback(accel_group, acceleratable, keyval, modifier):
            self.on_open_clicked(None)
            return True

        def process_callback(accel_group, acceleratable, keyval, modifier):
            self.on_process_clicked(None)
            return True

        def quit_callback(accel_group, acceleratable, keyval, modifier):
            Gtk.main_quit()
            return True

        def zoom_in_callback(accel_group, acceleratable, keyval, modifier):
            self.on_zoom_in_clicked(None)
            return True

        def zoom_out_callback(accel_group, acceleratable, keyval, modifier):
            self.on_zoom_out_clicked(None)
            return True

        def zoom_fit_callback(accel_group, acceleratable, keyval, modifier):
            self.on_zoom_fit_clicked(None)
            return True

        def zoom_100_callback(accel_group, acceleratable, keyval, modifier):
            self.on_zoom_100_clicked(None)
            return True

        # Ctrl+O for open file
        key, mod = Gtk.accelerator_parse("<Control>o")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, open_callback)

        # Ctrl+P for process
        key, mod = Gtk.accelerator_parse("<Control>p")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, process_callback)

        # Ctrl+Q for quit
        key, mod = Gtk.accelerator_parse("<Control>q")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, quit_callback)

        # Ctrl+F for fit to window
        key, mod = Gtk.accelerator_parse("<Control>f")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_fit_callback)

        # Ctrl++ for zoom in
        key, mod = Gtk.accelerator_parse("<Control>plus")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_in_callback)

        # Ctrl+= for zoom in (same key as + on US keyboard without shift)
        key, mod = Gtk.accelerator_parse("<Control>equal")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_in_callback)

        # Ctrl+KP_Add for zoom in (keypad +)
        key, mod = Gtk.accelerator_parse("<Control>KP_Add")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_in_callback)

        # Ctrl+- for zoom out
        key, mod = Gtk.accelerator_parse("<Control>minus")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_out_callback)

        # Ctrl+KP_Subtract for zoom out (keypad -)
        key, mod = Gtk.accelerator_parse("<Control>KP_Subtract")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_out_callback)

        # Ctrl+0 for 100% zoom
        key, mod = Gtk.accelerator_parse("<Control>0")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_100_callback)

        # Ctrl+KP_0 for 100% zoom (keypad 0)
        key, mod = Gtk.accelerator_parse("<Control>KP_0")
        accel_group.connect(key, mod, Gtk.AccelFlags.VISIBLE, zoom_100_callback)

        # Application state
        self.original_image: Optional[np.ndarray] = None
        self.current_file: Optional[Path] = None

        # Processing results
        self.warped_image: Optional[np.ndarray] = None
        self.detected_cols: Optional[int] = None
        self.detected_rows: Optional[int] = None
        self.grid_matrix: Optional[np.ndarray] = None

        # Visualization images
        self.vis_contours: Optional[np.ndarray] = None
        self.vis_lines: Optional[np.ndarray] = None
        self.vis_cells: Optional[np.ndarray] = None
        self.vis_dots: Optional[np.ndarray] = None

        # Zoom state
        self.zoom_level: float = 1.0  # Current zoom level (1.0 = 100%)
        self.current_image_cv: Optional[np.ndarray] = None  # Current displayed image (CV format)

        # Show the window
        self.window.show_all()
        self.update_status("Ready. Open an image to start.")

    def update_status(self, message: str):
        """Update the status bar."""
        context_id = self.statusbar.get_context_id("main")
        self.statusbar.pop(context_id)
        self.statusbar.push(context_id, message)

    def update_info(self, text: str):
        """Update the info text view."""
        buffer = self.text_info.get_buffer()
        buffer.set_text(text)

    def cv_to_pixbuf(self, cv_image: np.ndarray) -> GdkPixbuf.Pixbuf:
        """Convert OpenCV image (BGR) to GdkPixbuf."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape

        # Create pixbuf
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            rgb_image.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * channels,
        )
        return pixbuf

    def display_image(self, image: np.ndarray, reset_zoom: bool = False):
        """Display an image in the image display widget with current zoom level.

        Args:
            image: OpenCV image to display
            reset_zoom: If True, reset zoom to fit window
        """
        self.current_image_cv = image

        if reset_zoom:
            self.zoom_level = 1.0

        # Apply zoom
        if self.zoom_level != 1.0:
            height, width = image.shape[:2]
            new_width = int(width * self.zoom_level)
            new_height = int(height * self.zoom_level)
            zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            zoomed_image = image

        pixbuf = self.cv_to_pixbuf(zoomed_image)
        self.image_display.set_from_pixbuf(pixbuf)

        # Update status with zoom level
        zoom_percent = int(self.zoom_level * 100)
        current_status = self.statusbar.get_context_id("main")
        message = self.statusbar.get_message_area().get_children()[0].get_text()
        if " - Zoom:" in message:
            message = message.split(" - Zoom:")[0]
        self.update_status(f"{message} - Zoom: {zoom_percent}%")

    def on_window_destroy(self, widget):
        """Handle window close."""
        Gtk.main_quit()

    def on_quit_activate(self, widget=None):
        """Handle quit action."""
        Gtk.main_quit()

    def on_open_clicked(self, button):
        """Handle Open Image button click."""
        dialog = Gtk.FileChooserDialog(
            title="Open Crossword Image",
            parent=self.window,
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK
        )

        # Add image filter
        filter_image = Gtk.FileFilter()
        filter_image.set_name("Image files")
        filter_image.add_mime_type("image/jpeg")
        filter_image.add_mime_type("image/png")
        filter_image.add_pattern("*.jpg")
        filter_image.add_pattern("*.jpeg")
        filter_image.add_pattern("*.png")
        dialog.add_filter(filter_image)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            filename = dialog.get_filename()
            self.load_image(filename)

        dialog.destroy()

    def load_image(self, filename: str):
        """Load an image file."""
        try:
            self.current_file = Path(filename)
            self.original_image = cv2.imread(filename)

            if self.original_image is None:
                self.show_error("Failed to load image")
                return

            # Reset processing state
            self.warped_image = None
            self.detected_cols = None
            self.detected_rows = None
            self.grid_matrix = None
            self.vis_contours = None
            self.vis_lines = None
            self.vis_cells = None
            self.vis_dots = None

            # Display original image with zoom reset
            self.display_image(self.original_image, reset_zoom=True)
            self.radio_original.set_active(True)

            height, width = self.original_image.shape[:2]
            self.update_status(f"Loaded: {self.current_file.name} ({width}×{height})")
            self.update_info(f"Image: {self.current_file.name}\nSize: {width}×{height} pixels\n\nClick Process to analyze.")

        except Exception as e:
            self.show_error(f"Error loading image: {e}")

    def show_error(self, message: str):
        """Show an error dialog."""
        dialog = Gtk.MessageDialog(
            transient_for=self.window,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text="Error",
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def on_process_clicked(self, button):
        """Handle Process button click."""
        if self.original_image is None:
            self.show_error("Please open an image first")
            return

        self.update_status("Processing...")
        GLib.idle_add(self.process_image)

    def process_image(self):
        """Process the loaded image with current parameters."""
        try:
            # Get parameters
            threshold = int(self.scale_threshold.get_value())
            cell_margin = self.scale_cell_margin.get_value()
            dot_ratio = self.scale_dot_ratio.get_value()
            min_cell_size = int(self.scale_min_cell_size.get_value())
            detect_dots = self.check_detect_dots.get_active()

            # Step 1: Extract grid with contour visualization
            self.update_status("Step 1/4: Extracting grid...")
            self.vis_contours = self.visualize_contours(self.original_image)
            self.warped_image, width, height = extract_grid(self.original_image)

            # Step 2: Detect dimensions with line visualization
            self.update_status("Step 2/4: Detecting dimensions...")
            self.detected_cols, self.detected_rows = detect_grid_dimensions(self.warped_image)
            self.vis_lines = self.visualize_grid_lines(self.warped_image, self.detected_cols, self.detected_rows)

            # Step 3: Convert to matrix
            self.update_status("Step 3/4: Converting to matrix...")

            # Temporarily modify MIN_CELL_SIZE in the module
            import extract as extract_module
            original_min_cell = getattr(extract_module._detect_dot_in_cell, '__globals__', {}).get('MIN_CELL_SIZE', 30)

            # Patch the minimum cell size
            if hasattr(extract_module, '_detect_dot_in_cell'):
                # We'll need to recreate the function with new MIN_CELL_SIZE
                # For now, just call with current settings
                pass

            self.grid_matrix = convert_to_matrix(
                self.warped_image,
                width,
                height,
                self.detected_rows,
                self.detected_cols,
                intensity_threshold=threshold,
                cell_margin=cell_margin,
                detect_dots=detect_dots,
                dot_size_ratio=dot_ratio,
            )

            # Step 4: Create visualizations
            self.update_status("Step 4/4: Creating visualizations...")
            self.vis_cells = self.visualize_cells(self.warped_image, self.detected_cols, self.detected_rows, self.grid_matrix)
            self.vis_dots = self.visualize_dots(self.warped_image, self.detected_cols, self.detected_rows, self.grid_matrix)

            # Update info panel
            white_cells = np.sum(self.grid_matrix == 1)
            black_cells = np.sum(self.grid_matrix == 0)
            dot_cells = np.sum(self.grid_matrix == 2)

            cell_width = width / self.detected_cols
            cell_height = height / self.detected_rows

            info_text = f"""Image: {self.current_file.name}
Original size: {self.original_image.shape[1]}×{self.original_image.shape[0]}
Extracted grid: {width}×{height} pixels

Detected dimensions: {self.detected_cols}×{self.detected_rows}
Cell size: {cell_width:.1f}×{cell_height:.1f} pixels

Grid statistics:
  White cells: {white_cells}
  Black cells: {black_cells}
  Cells with dots: {dot_cells}

Parameters:
  Intensity threshold: {threshold}
  Cell margin: {cell_margin:.2f}
  Dot size ratio: {dot_ratio:.2f}
  Min cell size: {min_cell_size}px
  Dot detection: {"Enabled" if detect_dots else "Disabled"}
"""
            self.update_info(info_text)
            self.update_status("Processing complete!")

            # Update current view
            self.update_display()

        except GridExtractionError as e:
            self.show_error(f"Grid extraction failed: {e}")
            self.update_status("Error: Grid extraction failed")
        except DimensionDetectionError as e:
            self.show_error(f"Dimension detection failed: {e}")
            self.update_status("Error: Dimension detection failed")
        except Exception as e:
            self.show_error(f"Processing failed: {e}")
            self.update_status("Error: Processing failed")

    def visualize_contours(self, image: np.ndarray) -> np.ndarray:
        """Create visualization of contour detection."""
        vis = image.copy()

        # Apply thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw all contours in blue
        cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)

        # Find and highlight the largest quadrilateral
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

            if len(approx) == 4:
                # Draw the detected grid contour in green
                cv2.drawContours(vis, [approx], -1, (0, 255, 0), 3)

        return vis

    def visualize_grid_lines(self, warped: np.ndarray, cols: int, rows: int) -> np.ndarray:
        """Create visualization of detected grid lines."""
        vis = warped.copy()
        height, width = warped.shape[:2]

        cell_width = width / cols
        cell_height = height / rows

        # Draw vertical lines (columns)
        for i in range(cols + 1):
            x = int(i * cell_width)
            cv2.line(vis, (x, 0), (x, height), (0, 255, 0), 1)

        # Draw horizontal lines (rows)
        for i in range(rows + 1):
            y = int(i * cell_height)
            cv2.line(vis, (0, y), (width, y), (0, 255, 0), 1)

        return vis

    def visualize_cells(self, warped: np.ndarray, cols: int, rows: int, matrix: np.ndarray) -> np.ndarray:
        """Create visualization of cell classification."""
        height, width = warped.shape[:2]

        # Create a blank canvas
        vis = np.zeros((height, width, 3), dtype=np.uint8)

        cell_width = width / cols
        cell_height = height / rows

        for r in range(rows):
            for c in range(cols):
                x1 = int(c * cell_width)
                y1 = int(r * cell_height)
                x2 = int((c + 1) * cell_width)
                y2 = int((r + 1) * cell_height)

                # Color based on classification
                if matrix[r, c] == 0:
                    # Black cell - dark gray
                    color = (50, 50, 50)
                elif matrix[r, c] == 1:
                    # White cell - white
                    color = (255, 255, 255)
                else:  # matrix[r, c] == 2
                    # White cell with dot - light green
                    color = (200, 255, 200)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (128, 128, 128), 1)

        return vis

    def visualize_dots(self, warped: np.ndarray, cols: int, rows: int, matrix: np.ndarray) -> np.ndarray:
        """Create visualization highlighting dots."""
        vis = warped.copy()
        height, width = warped.shape[:2]

        cell_width = width / cols
        cell_height = height / rows

        for r in range(rows):
            for c in range(cols):
                if matrix[r, c] == 2:  # Cell with dot
                    x = int((c + 0.5) * cell_width)
                    y = int((r + 0.5) * cell_height)

                    # Draw a red circle at the cell center
                    cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

                    # Draw the dot detection region (bottom-right corner)
                    dot_region_w = int(cell_width * 0.2)
                    dot_region_h = int(cell_height * 0.2)
                    x1 = int((c + 1) * cell_width - dot_region_w)
                    y1 = int((r + 1) * cell_height - dot_region_h)
                    x2 = int((c + 1) * cell_width)
                    y2 = int((r + 1) * cell_height)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

        return vis

    def on_view_changed(self, radio_button):
        """Handle view selection change."""
        if radio_button.get_active():
            self.update_display()

    def update_display(self):
        """Update the displayed image based on selected view."""
        if self.radio_original.get_active():
            if self.original_image is not None:
                self.display_image(self.original_image)
        elif self.radio_contours.get_active():
            if self.vis_contours is not None:
                self.display_image(self.vis_contours)
        elif self.radio_extracted.get_active():
            if self.warped_image is not None:
                self.display_image(self.warped_image)
        elif self.radio_lines.get_active():
            if self.vis_lines is not None:
                self.display_image(self.vis_lines)
        elif self.radio_cells.get_active():
            if self.vis_cells is not None:
                self.display_image(self.vis_cells)
        elif self.radio_dots.get_active():
            if self.vis_dots is not None:
                self.display_image(self.vis_dots)

    def on_parameter_changed(self, widget):
        """Handle parameter change - reprocess if image is loaded."""
        if self.original_image is not None and self.warped_image is not None:
            # Auto-reprocess when parameters change
            GLib.timeout_add(500, self.process_image)  # Debounce: wait 500ms

    def on_zoom_in_clicked(self, button):
        """Handle Zoom In button click."""
        if self.current_image_cv is not None:
            # Zoom in by 25%
            self.zoom_level *= 1.25
            # Cap at 500%
            if self.zoom_level > 5.0:
                self.zoom_level = 5.0
            self.display_image(self.current_image_cv)

    def on_zoom_out_clicked(self, button):
        """Handle Zoom Out button click."""
        if self.current_image_cv is not None:
            # Zoom out by 25%
            self.zoom_level /= 1.25
            # Cap at 10%
            if self.zoom_level < 0.1:
                self.zoom_level = 0.1
            self.display_image(self.current_image_cv)

    def on_zoom_fit_clicked(self, button):
        """Handle Fit to Window button click."""
        if self.current_image_cv is not None:
            # Get the scrolled window size
            scrolled_window = self.image_display.get_parent()
            allocation = scrolled_window.get_allocation()
            window_width = allocation.width
            window_height = allocation.height

            # Get image size
            img_height, img_width = self.current_image_cv.shape[:2]

            # Calculate zoom to fit
            zoom_w = window_width / img_width
            zoom_h = window_height / img_height
            self.zoom_level = min(zoom_w, zoom_h) * 0.95  # 95% to leave some margin

            self.display_image(self.current_image_cv)

    def on_zoom_100_clicked(self, button):
        """Handle 100% Zoom button click."""
        if self.current_image_cv is not None:
            self.zoom_level = 1.0
            self.display_image(self.current_image_cv)


def main():
    """Main entry point."""
    app = CrosswordVisualizer()

    # If an image file is provided as argument, load it
    if len(sys.argv) > 1:
        app.load_image(sys.argv[1])

    Gtk.main()


if __name__ == "__main__":
    main()
