#!/usr/bin/env bash
#
# Launch the Crossword Grid Extraction Visualizer
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default image if none provided
DEFAULT_IMAGE="$SCRIPT_DIR/../test_data/wciska_kig.jpg"

# Use provided image or default
IMAGE="${1:-$DEFAULT_IMAGE}"

# Run the visualizer
exec python "$SCRIPT_DIR/src/visualizer.py" "$IMAGE"
