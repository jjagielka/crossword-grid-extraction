#!/bin/bash
# Installation script for Crossword MCP Server systemd service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_FILE="$PROJECT_DIR/systemd/crossword-mcp.service"
SERVICE_TEMPLATE="$SERVICE_FILE"
SYSTEMD_DIR="$HOME/.config/systemd/user"

echo -e "${GREEN}Crossword MCP Server - Systemd Installation${NC}"
echo "=============================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}Warning: This script should not be run as root.${NC}"
    echo "It will install the service for the current user: $USER"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Python is available
echo -e "${YELLOW}[1/6]${NC} Detecting Python installation..."
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found in system PATH.${NC}"
    echo "  Please install Python 3.11+ first."
    exit 1
fi
echo "  Python found: $(python --version)"

# Check if fastmcp is installed
echo -e "${YELLOW}[2/6]${NC} Checking dependencies..."
if ! python -c "import fastmcp" 2>/dev/null; then
    echo -e "${RED}Error: fastmcp is not installed.${NC}"
    echo ""
    echo "Please install MCP dependencies:"
    echo "  uv pip install -e '.[mcp]'"
    exit 1
fi
echo "  All dependencies installed ✓"

# Check if mcp_server.py exists
echo -e "${YELLOW}[3/6]${NC} Verifying MCP server..."
if [ ! -f "$PROJECT_DIR/src/mcp_server.py" ]; then
    echo -e "${RED}Error: mcp_server.py not found at $PROJECT_DIR/src/${NC}"
    exit 1
fi
echo "  MCP server found ✓"

# Create systemd user directory if it doesn't exist
echo -e "${YELLOW}[4/6]${NC} Setting up systemd user directory..."
mkdir -p "$SYSTEMD_DIR"
echo "  Directory created: $SYSTEMD_DIR"

# Create the service file with substitutions
echo -e "${YELLOW}[5/6]${NC} Installing systemd service..."
TEMP_SERVICE=$(mktemp)

# Detect user's primary group
USER_GROUP=$(id -gn)

# Replace placeholders in the service file
sed -e "s|%USER%|$USER|g" \
    -e "s|%GROUP%|$USER_GROUP|g" \
    -e "s|%PROJECT_DIR%|$PROJECT_DIR|g" \
    "$SERVICE_TEMPLATE" > "$TEMP_SERVICE"

# Install the service file
cp "$TEMP_SERVICE" "$SYSTEMD_DIR/crossword-mcp.service"
rm "$TEMP_SERVICE"

echo "  Service file installed: $SYSTEMD_DIR/crossword-mcp.service"

# Reload systemd
echo -e "${YELLOW}[6/6]${NC} Reloading systemd daemon..."
systemctl --user daemon-reload
echo "  Systemd daemon reloaded ✓"

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Start the service:"
echo "     systemctl --user start crossword-mcp"
echo ""
echo "  2. Check status:"
echo "     systemctl --user status crossword-mcp"
echo ""
echo "  3. Enable auto-start on boot:"
echo "     systemctl --user enable crossword-mcp"
echo "     loginctl enable-linger $USER"
echo ""
echo "  4. View logs:"
echo "     journalctl --user -u crossword-mcp -f"
echo ""
echo "  5. Stop the service:"
echo "     systemctl --user stop crossword-mcp"
echo ""
echo "  6. Restart the service:"
echo "     systemctl --user restart crossword-mcp"
echo ""
echo "Useful commands:"
echo "  - Disable the service: systemctl --user disable crossword-mcp"
echo "  - Uninstall: rm $SYSTEMD_DIR/crossword-mcp.service && systemctl --user daemon-reload"
echo ""
