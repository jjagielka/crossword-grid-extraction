#!/usr/bin/env python3
"""
Crossword Grid Extraction Visualizer (GTK4)

A GTK4 application for visualizing the detection steps and tuning parameters.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, Gio
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Optional

# Import extraction functions
from extract import (
    extract_grid,
    detect_grid_dimensions,
    convert_to_matrix,
    GridExtractionError,
    DimensionDetectionError,
)


class CrosswordVisualizerApp(Gtk.Application):
    """Main GTK4 application for crossword grid extraction visualization."""

    def __init__(self):
        super().__init__(application_id="com.crossword.visualizer", flags=Gio.ApplicationFlags.HANDLES_OPEN)
        self.window = None
        self.setup_actions()

    def setup_actions(self):
        """Setup application actions and keyboard shortcuts in a data-driven way."""
        # Command actions: (name, callback, accelerators)
        commands = [
            ("open", self.on_command, ["<Primary>o"]),
            ("process", self.on_command, ["<Primary>p"]),
            ("zoom-in", self.on_command, ["<Primary>plus", "<Primary>equal", "<Primary>KP_Add"]),
            ("zoom-out", self.on_command, ["<Primary>minus", "<Primary>KP_Subtract"]),
            ("zoom-fit", self.on_command, ["<Primary>f"]),
            ("zoom-100", self.on_command, ["<Primary>0", "<Primary>KP_0"]),
            ("quit", self.on_command, ["<Primary>q"]),
        ]

        for name, callback, accels in commands:
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", callback)
            self.add_action(action)
            self.set_accels_for_action(f"app.{name}", accels)

        # Stateful actions: (name, type, default_value, callback)
        stateful = [
            ("view", "s", "original", self.on_state_change),
            ("threshold", "d", 140.0, self.on_parameter_change),
            ("cell-margin", "d", 0.25, self.on_parameter_change),
            ("dot-ratio", "d", 0.20, self.on_parameter_change),
            ("min-cell-size", "d", 30.0, self.on_parameter_change),
            ("detect-dots", "b", True, self.on_parameter_change),
            ("use-curved-lines", "b", True, self.on_parameter_change),
            ("curve-smoothing", "d", 100.0, self.on_parameter_change),
        ]

        for name, vtype, default, callback in stateful:
            if vtype == "b":  # boolean
                action = Gio.SimpleAction.new_stateful(name, None, GLib.Variant("b", default))
            else:  # string or double
                action = Gio.SimpleAction.new_stateful(
                    name, GLib.VariantType.new(vtype), GLib.Variant(vtype, default)
                )
            action.connect("change-state", callback)
            self.add_action(action)

    def on_command(self, action, param):
        """Generic handler for command actions - dispatches to window methods."""
        if not self.window:
            return

        # Map action names to window methods
        action_map = {
            "open": "on_open_clicked",
            "process": "on_process_clicked",
            "zoom-in": "on_zoom_in_clicked",
            "zoom-out": "on_zoom_out_clicked",
            "zoom-fit": "on_zoom_fit_clicked",
            "zoom-100": "on_zoom_100_clicked",
        }

        action_name = action.get_name()
        if action_name == "quit":
            self.quit()
        elif method_name := action_map.get(action_name):
            getattr(self.window, method_name)()

    def on_state_change(self, action, value):
        """Generic handler for stateful actions that change views."""
        action.set_state(value)
        if self.window and action.get_name() == "view":
            self.window.switch_view(value.get_string())

    def on_parameter_change(self, action, value):
        """Generic handler for parameter changes - updates state and triggers reprocessing."""
        action.set_state(value)
        if self.window:
            self.window.on_parameter_changed(None)

    def do_activate(self):
        """Called when the application is activated."""
        if not self.window:
            self.window = CrosswordVisualizerWindow(application=self)
        self.window.present()

    def do_open(self, files, n_files, hint):
        """Called when files are opened."""
        self.do_activate()
        if n_files > 0:
            file_path = files[0].get_path()
            self.window.load_image(file_path)


class CrosswordVisualizerWindow(Gtk.ApplicationWindow):
    """Main application window."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load UI from file
        builder = Gtk.Builder()
        ui_file = Path(__file__).parent / "visualizer_gtk4.ui"

        # Load the UI file
        builder.add_from_file(str(ui_file))

        # Get the main content box and set it as our child
        main_content = builder.get_object("main_content")
        self.set_child(main_content)
        self.set_default_size(1200, 800)
        self.set_title("Crossword Grid Extraction Visualizer")

        # Get widgets
        self.image_display = builder.get_object("image_display")
        self.text_info = builder.get_object("text_info")

        # Get controls
        self.scale_threshold = builder.get_object("scale_threshold")
        self.scale_cell_margin = builder.get_object("scale_cell_margin")
        self.scale_dot_ratio = builder.get_object("scale_dot_ratio")
        self.scale_min_cell_size = builder.get_object("scale_min_cell_size")
        self.check_detect_dots = builder.get_object("check_detect_dots")
        self.check_use_curved_lines = builder.get_object("check_use_curved_lines")
        self.scale_curve_smoothing = builder.get_object("scale_curve_smoothing")

        # Get radio buttons
        self.radio_original = builder.get_object("radio_original")
        self.radio_contours = builder.get_object("radio_contours")
        self.radio_extracted = builder.get_object("radio_extracted")
        self.radio_lines = builder.get_object("radio_lines")
        self.radio_curved_lines = builder.get_object("radio_curved_lines")
        self.radio_cells = builder.get_object("radio_cells")
        self.radio_dots = builder.get_object("radio_dots")

        # Connect scale signals manually (Python approach)
        # This is more reliable than BuilderScope for bound methods
        self.scale_threshold.connect("value-changed", self.on_threshold_changed)
        self.scale_cell_margin.connect("value-changed", self.on_margin_changed)
        self.scale_dot_ratio.connect("value-changed", self.on_dot_ratio_changed)
        self.scale_min_cell_size.connect("value-changed", self.on_min_cell_changed)
        self.scale_curve_smoothing.connect("value-changed", self.on_curve_smoothing_changed)

        # Note: Toolbar buttons, radio buttons, and checkboxes use action-name property
        # and are connected automatically via the GTK action system

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
        self.vis_curved_lines: Optional[np.ndarray] = None
        self.vis_cells: Optional[np.ndarray] = None
        self.vis_dots: Optional[np.ndarray] = None

        # Splines for curved grid lines
        self.horizontal_splines = None
        self.vertical_splines = None

        # Zoom state
        self.zoom_level: float = 1.0
        self.current_image_cv: Optional[np.ndarray] = None

        # Setup keyboard shortcuts
        # Note: Keyboard shortcuts are now defined via GAction in the Application class
        # and set_accels_for_action() calls, not with GtkShortcutController

        self.update_info("Ready. Open an image to start.")

    def update_info(self, text: str):
        """Update the info text view."""
        buffer = self.text_info.get_buffer()
        buffer.set_text(text, -1)

    def cv_to_texture(self, cv_image: np.ndarray) -> Gdk.Texture:
        """Convert OpenCV image (BGR) to Gdk.Texture for GTK4."""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape

        # Create GdkPixbuf first
        pixbuf = GdkPixbuf.Pixbuf.new_from_data(
            rgb_image.tobytes(),
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            width,
            height,
            width * channels,
        )

        # Convert to Gdk.Texture
        return Gdk.Texture.new_for_pixbuf(pixbuf)

    def display_image(self, image: np.ndarray, reset_zoom: bool = False):
        """Display an image with current zoom level."""
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

        texture = self.cv_to_texture(zoomed_image)
        self.image_display.set_paintable(texture)

    def on_open_clicked(self, *args):
        """Handle Open Image button click."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Open Crossword Image")

        # Create file filter
        filter_image = Gtk.FileFilter()
        filter_image.set_name("Image files")
        filter_image.add_mime_type("image/jpeg")
        filter_image.add_mime_type("image/png")
        filter_image.add_pattern("*.jpg")
        filter_image.add_pattern("*.jpeg")
        filter_image.add_pattern("*.png")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(filter_image)
        dialog.set_filters(filters)

        dialog.open(self, None, self.on_open_dialog_response)

    def on_open_dialog_response(self, dialog, result):
        """Handle file dialog response."""
        try:
            file = dialog.open_finish(result)
            if file:
                filename = file.get_path()
                self.load_image(filename)
        except Exception as e:
            # User cancelled
            pass

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

            # Display original image
            self.display_image(self.original_image, reset_zoom=True)
            self.radio_original.set_active(True)

            height, width = self.original_image.shape[:2]
            self.update_info(
                f"Image: {self.current_file.name}\nSize: {width}×{height} pixels\n\nClick Process to analyze."
            )

        except Exception as e:
            self.show_error(f"Error loading image: {e}")

    def show_error(self, message: str):
        """Show an error dialog."""
        dialog = Gtk.AlertDialog()
        dialog.set_message("Error")
        dialog.set_detail(message)
        dialog.set_buttons(["OK"])
        dialog.choose(self, None, None)

    def on_process_clicked(self, *args):
        """Handle Process button click."""
        if self.original_image is None:
            self.show_error("Please open an image first")
            return

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
            use_curved_lines = self.check_use_curved_lines.get_active()
            curve_smoothing = self.scale_curve_smoothing.get_value()

            # Step 1: Extract grid
            self.vis_contours = self.visualize_contours(self.original_image)
            self.warped_image, width, height = extract_grid(self.original_image)

            # Step 2: Detect dimensions
            self.detected_cols, self.detected_rows = detect_grid_dimensions(self.warped_image)
            self.vis_lines = self.visualize_grid_lines(
                self.warped_image, self.detected_cols, self.detected_rows
            )

            # Step 3: Convert to matrix (with curved grid line detection)
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
                use_curved_lines=use_curved_lines,
                curve_smoothing=curve_smoothing,
            )

            # Detect curved grid lines for visualization (if enabled)
            if use_curved_lines:
                from extract import detect_curved_grid_lines

                self.horizontal_splines, self.vertical_splines = detect_curved_grid_lines(
                    self.warped_image,
                    self.detected_rows,
                    self.detected_cols,
                    smoothing_factor=curve_smoothing,
                )
                # Step 4: Create curved line visualization
                self.vis_curved_lines = self.visualize_curved_grid_lines(
                    self.warped_image, self.horizontal_splines, self.vertical_splines
                )
            else:
                self.vis_curved_lines = None

            # Step 4: Create other visualizations
            self.vis_cells = self.visualize_cells(
                self.warped_image, self.detected_cols, self.detected_rows, self.grid_matrix
            )
            self.vis_dots = self.visualize_dots(
                self.warped_image, self.detected_cols, self.detected_rows, self.grid_matrix
            )

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
  Curved detection: {"Enabled" if use_curved_lines else "Disabled"}
  Curve smoothing: {curve_smoothing:.0f}
"""
            self.update_info(info_text)
            self.update_display()

        except GridExtractionError as e:
            self.show_error(f"Grid extraction failed: {e}")
        except DimensionDetectionError as e:
            self.show_error(f"Dimension detection failed: {e}")
        except Exception as e:
            self.show_error(f"Processing failed: {e}")

    def visualize_contours(self, image: np.ndarray) -> np.ndarray:
        """Create visualization of contour detection."""
        vis = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                cv2.drawContours(vis, [approx], -1, (0, 255, 0), 3)
        return vis

    def visualize_grid_lines(self, warped: np.ndarray, cols: int, rows: int) -> np.ndarray:
        """Create visualization of detected grid lines."""
        vis = warped.copy()
        height, width = warped.shape[:2]
        cell_width = width / cols
        cell_height = height / rows

        for i in range(cols + 1):
            x = int(i * cell_width)
            cv2.line(vis, (x, 0), (x, height), (0, 255, 0), 1)

        for i in range(rows + 1):
            y = int(i * cell_height)
            cv2.line(vis, (0, y), (width, y), (0, 255, 0), 1)

        return vis

    def visualize_curved_grid_lines(self, warped: np.ndarray, h_splines, v_splines) -> np.ndarray:
        """Create visualization of curved grid lines detected via spline fitting."""
        vis = warped.copy()
        height, width = warped.shape[:2]

        # Draw horizontal splines (green)
        for spline in h_splines:
            x_coords = np.linspace(0, width - 1, 100, dtype=int)
            y_coords = spline(x_coords).astype(int)
            # Clip to valid range
            y_coords = np.clip(y_coords, 0, height - 1)
            points = np.column_stack([x_coords, y_coords])
            cv2.polylines(vis, [points], False, (0, 255, 0), 2)

        # Draw vertical splines (blue)
        for spline in v_splines:
            y_coords = np.linspace(0, height - 1, 100, dtype=int)
            x_coords = spline(y_coords).astype(int)
            # Clip to valid range
            x_coords = np.clip(x_coords, 0, width - 1)
            points = np.column_stack([x_coords, y_coords])
            cv2.polylines(vis, [points], False, (255, 0, 0), 2)

        return vis

    def visualize_cells(self, warped: np.ndarray, cols: int, rows: int, matrix: np.ndarray) -> np.ndarray:
        """Create visualization of cell classification."""
        height, width = warped.shape[:2]
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        cell_width = width / cols
        cell_height = height / rows

        for r in range(rows):
            for c in range(cols):
                x1 = int(c * cell_width)
                y1 = int(r * cell_height)
                x2 = int((c + 1) * cell_width)
                y2 = int((r + 1) * cell_height)

                if matrix[r, c] == 0:
                    color = (50, 50, 50)
                elif matrix[r, c] == 1:
                    color = (255, 255, 255)
                else:
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
                if matrix[r, c] == 2:
                    x = int((c + 0.5) * cell_width)
                    y = int((r + 0.5) * cell_height)
                    cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)

                    dot_region_w = int(cell_width * 0.2)
                    dot_region_h = int(cell_height * 0.2)
                    x1 = int((c + 1) * cell_width - dot_region_w)
                    y1 = int((r + 1) * cell_height - dot_region_h)
                    x2 = int((c + 1) * cell_width)
                    y2 = int((r + 1) * cell_height)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

        return vis

    def switch_view(self, view_name: str):
        """Switch to a different visualization view (called from action)."""
        view_map = {
            "original": (self.original_image, self.radio_original),
            "contours": (self.vis_contours, self.radio_contours),
            "extracted": (self.warped_image, self.radio_extracted),
            "lines": (self.vis_lines, self.radio_lines),
            "curved_lines": (self.vis_curved_lines, self.radio_curved_lines),
            "cells": (self.vis_cells, self.radio_cells),
            "dots": (self.vis_dots, self.radio_dots),
        }

        image, radio = view_map.get(view_name, (None, None))
        if image is not None:
            self.display_image(image)
            # Update radio button to match (sync UI with action state)
            if radio:
                radio.set_active(True)

    def update_display(self):
        """Update the displayed image based on selected view."""
        if self.radio_original.get_active() and self.original_image is not None:
            self.display_image(self.original_image)
        elif self.radio_contours.get_active() and self.vis_contours is not None:
            self.display_image(self.vis_contours)
        elif self.radio_extracted.get_active() and self.warped_image is not None:
            self.display_image(self.warped_image)
        elif self.radio_lines.get_active() and self.vis_lines is not None:
            self.display_image(self.vis_lines)
        elif self.radio_curved_lines.get_active() and self.vis_curved_lines is not None:
            self.display_image(self.vis_curved_lines)
        elif self.radio_cells.get_active() and self.vis_cells is not None:
            self.display_image(self.vis_cells)
        elif self.radio_dots.get_active() and self.vis_dots is not None:
            self.display_image(self.vis_dots)

    def on_parameter_changed(self, widget):
        """Handle parameter change - reprocess if image is loaded."""
        if self.original_image is not None and self.warped_image is not None:
            GLib.timeout_add(500, self.process_image)

    def on_threshold_changed(self, scale):
        """Handle threshold scale change - update action state."""
        value = scale.get_value()
        if action := self.get_application().lookup_action("threshold"):
            action.set_state(GLib.Variant("d", value))
        self.on_parameter_changed(scale)

    def on_margin_changed(self, scale):
        """Handle cell margin scale change - update action state."""
        value = scale.get_value()
        if action := self.get_application().lookup_action("cell-margin"):
            action.set_state(GLib.Variant("d", value))
        self.on_parameter_changed(scale)

    def on_dot_ratio_changed(self, scale):
        """Handle dot ratio scale change - update action state."""
        value = scale.get_value()
        if action := self.get_application().lookup_action("dot-ratio"):
            action.set_state(GLib.Variant("d", value))
        self.on_parameter_changed(scale)

    def on_min_cell_changed(self, scale):
        """Handle min cell size scale change - update action state."""
        value = scale.get_value()
        if action := self.get_application().lookup_action("min-cell-size"):
            action.set_state(GLib.Variant("d", value))
        self.on_parameter_changed(scale)

    def on_curve_smoothing_changed(self, scale):
        """Handle curve smoothing scale change - update action state."""
        value = scale.get_value()
        if action := self.get_application().lookup_action("curve-smoothing"):
            action.set_state(GLib.Variant("d", value))
        self.on_parameter_changed(scale)

    def on_zoom_in_clicked(self, *args):
        """Handle Zoom In."""
        if self.current_image_cv is not None:
            self.zoom_level *= 1.25
            if self.zoom_level > 5.0:
                self.zoom_level = 5.0
            self.display_image(self.current_image_cv)

    def on_zoom_out_clicked(self, *args):
        """Handle Zoom Out."""
        if self.current_image_cv is not None:
            self.zoom_level /= 1.25
            if self.zoom_level < 0.1:
                self.zoom_level = 0.1
            self.display_image(self.current_image_cv)

    def on_zoom_fit_clicked(self, *args):
        """Handle Fit to Window."""
        if self.current_image_cv is not None:
            # Get scrolled window allocation
            parent = self.image_display.get_parent()
            width = parent.get_width()
            height = parent.get_height()

            img_height, img_width = self.current_image_cv.shape[:2]
            zoom_w = width / img_width
            zoom_h = height / img_height
            self.zoom_level = min(zoom_w, zoom_h) * 0.95

            self.display_image(self.current_image_cv)

    def on_zoom_100_clicked(self, *args):
        """Handle 100% Zoom."""
        if self.current_image_cv is not None:
            self.zoom_level = 1.0
            self.display_image(self.current_image_cv)


def main():
    """Main entry point."""
    app = CrosswordVisualizerApp()

    # Pass command-line arguments to GTK - it will call do_open if files are provided
    # or do_activate if no files are provided
    app.run(sys.argv)


if __name__ == "__main__":
    main()
