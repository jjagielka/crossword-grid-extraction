# Quick Start: Systemd Deployment

Deploy the Crossword MCP Server as a systemd service on Linux.

## Prerequisites

```bash
cd /home/jakub/crossword-read
uv pip install -e '.[mcp]'
```

## Installation (One-Time Setup)

```bash
./systemd/install.sh
```

## Basic Commands

```bash
# Start the service
systemctl --user start crossword-mcp

# Check status
systemctl --user status crossword-mcp

# View logs (live)
journalctl --user -u crossword-mcp -f

# Stop the service
systemctl --user stop crossword-mcp

# Restart the service
systemctl --user restart crossword-mcp
```

## Enable Auto-Start on Boot

```bash
systemctl --user enable crossword-mcp
loginctl enable-linger $USER
```

## Test the Service

```bash
# Check if it's running
curl http://127.0.0.1:8000/health

# View SSE endpoint
curl http://127.0.0.1:8000/sse
```

## Troubleshooting

```bash
# View recent logs
journalctl --user -u crossword-mcp -n 50

# Check service file
systemctl --user cat crossword-mcp

# Verify dependencies
python -c "import fastmcp; import cv2"
```

## What's Running?

- **Mode:** HTTP/SSE transport
- **Port:** 8000
- **URL:** `http://127.0.0.1:8000/sse`
- **Transport:** FastMCP with uvicorn (built-in)

## Important Notes

- This runs the server in **HTTP/SSE mode**, not STDIO mode
- For Claude Desktop, use STDIO mode instead (direct Python execution)
- See `docs/DEPLOYMENT_OPTIONS.md` for choosing the right deployment
- Full documentation: `docs/SYSTEMD_DEPLOYMENT.md`

## Uninstall

```bash
systemctl --user stop crossword-mcp
systemctl --user disable crossword-mcp
rm ~/.config/systemd/user/crossword-mcp.service
systemctl --user daemon-reload
```
