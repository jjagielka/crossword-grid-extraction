# Examples

This folder contains example scripts and configuration files for using the crossword extraction tool.

## Configuration Files

### claude_desktop_config.json

Example configuration for integrating the MCP server with Claude Desktop.

To use:
1. Copy this file's contents to your Claude Desktop config location:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`

2. Update the absolute path to point to your `mcp_server.py` location

3. Restart Claude Desktop

## Future Examples

Planned examples:
- Batch processing multiple crosswords
- Custom threshold tuning
- Integration with OCR for clue extraction
- Visualization of detection steps
- Converting to different output formats

## Basic CLI Usage

For CLI usage examples, refer to the main README.md.

Basic commands:
```bash
# Extract grid
python src/crossword.py --input=crossword.jpg extract

# Detect dimensions
python src/crossword.py --input=crossword.jpg size

# Full conversion
python src/crossword.py --input=crossword.jpg convert
```

## Contributing Examples

If you create useful example scripts, please add them here with documentation.
