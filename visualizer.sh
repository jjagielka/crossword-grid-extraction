#!/bin/bash
# Launch script for the Crossword Grid Extraction Visualizer

cd "$(dirname "$0")"
python src/visualizer.py "$@"
