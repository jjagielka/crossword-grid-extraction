# Systemd Deployment Guide

This guide explains how to deploy the Crossword MCP Server as a systemd service on Linux.

## Important Note: MCP Transport Modes

The MCP server supports two transport modes:

1. **STDIO mode** (default): Used by Claude Desktop and similar clients
   - Server launched on-demand by the client
   - Communicates via stdin/stdout pipes
   - **NOT suitable for systemd daemon** (no interactive client)
   - Usage: `python src/mcp_server.py`

2. **HTTP/SSE mode**: For persistent server deployment
   - Server runs as HTTP service on port 8000
   - **This is what systemd deployment uses**
   - FastMCP includes uvicorn internally
   - Usage: `python src/mcp_server.py --http`
   - Server URL: `http://127.0.0.1:8000/sse`

**This guide covers HTTP/SSE deployment with systemd.** For STDIO mode (Claude Desktop), clients should launch the server directly without systemd.

## Overview

Running as a systemd service provides:
- Automatic startup on boot (with `enable-linger`)
- Automatic restart on failure
- Centralized logging via journald
- Resource limits and security hardening
- Easy management via `systemctl` commands

## Prerequisites

1. **Linux system with systemd** (most modern distributions)
2. **Python 3.11+** installed
3. **Project dependencies installed** including FastMCP

## Quick Start

### 1. Install Dependencies

First, ensure you have all dependencies:

```bash
cd /home/jakub/crossword-read

# Install the project with MCP dependencies
uv pip install -e '.[mcp]'
```

### 2. Run the Installation Script

```bash
./systemd/install.sh
```

The script will:
- Check Python installation
- Verify all dependencies are installed
- Create the systemd user directory (`~/.config/systemd/user/`)
- Install the service file with correct paths
- Reload the systemd daemon

### 3. Start the Service

```bash
# Start the service
systemctl --user start crossword-mcp

# Check status
systemctl --user status crossword-mcp
```

### 4. Enable Auto-Start on Boot (Optional)

```bash
# Enable the service to start automatically
systemctl --user enable crossword-mcp

# Allow the service to run even when not logged in
loginctl enable-linger $USER
```

## Service Management

### Basic Commands

```bash
# Start the service
systemctl --user start crossword-mcp

# Stop the service
systemctl --user stop crossword-mcp

# Restart the service
systemctl --user restart crossword-mcp

# Check status
systemctl --user status crossword-mcp

# Enable auto-start on boot
systemctl --user enable crossword-mcp

# Disable auto-start
systemctl --user disable crossword-mcp
```

### Viewing Logs

```bash
# View recent logs
journalctl --user -u crossword-mcp

# Follow logs in real-time
journalctl --user -u crossword-mcp -f

# View logs since last boot
journalctl --user -u crossword-mcp -b

# View logs from the last hour
journalctl --user -u crossword-mcp --since "1 hour ago"

# View logs with line numbers
journalctl --user -u crossword-mcp -n 100
```

### Troubleshooting

**Service won't start:**
```bash
# Check detailed status
systemctl --user status crossword-mcp

# View recent logs
journalctl --user -u crossword-mcp -n 50

# Verify the service file syntax
systemd-analyze --user verify crossword-mcp.service
```

**Service keeps restarting:**
- Check the logs for errors
- Verify Python and dependencies are correctly installed
- Check file permissions on `mcp_server.py`

**Can't find the service:**
```bash
# Reload systemd daemon
systemctl --user daemon-reload

# List all user services
systemctl --user list-units --type=service
```

## Service Configuration

The service file is located at: `~/.config/systemd/user/crossword-mcp.service`

### Key Settings

**Restart Policy:**
- Automatic restart on failure
- 10-second delay between restarts
- Maximum 3 restart attempts per minute

**Resource Limits:**
- Memory: 1GB maximum
- CPU: 50% quota

**Security Hardening:**
- `NoNewPrivileges=true` - Prevents privilege escalation
- `PrivateTmp=true` - Isolated /tmp directory
- `ProtectSystem=strict` - Read-only system directories
- `ProtectHome=read-only` - Read-only home directory (except /tmp)

### Modifying the Service

If you need to modify the service configuration:

1. Edit the template file:
   ```bash
   nano /home/jakub/crossword-read/systemd/crossword-mcp.service
   ```

2. Re-run the installation script:
   ```bash
   ./systemd/install.sh
   ```

3. Reload and restart:
   ```bash
   systemctl --user daemon-reload
   systemctl --user restart crossword-mcp
   ```

## Using with MCP Clients

### HTTP/SSE Mode (with systemd)

When running as a systemd service in HTTP/SSE mode, the server is accessible at:
```
http://127.0.0.1:8000/sse
```

MCP clients that support HTTP/SSE transport can connect to this URL.

**Testing the HTTP endpoint:**
```bash
# Check if server is responding
curl http://127.0.0.1:8000/health

# View SSE endpoint (for debugging)
curl http://127.0.0.1:8000/sse
```

### STDIO Mode (without systemd - recommended for Claude Desktop)

For Claude Desktop on Linux, use STDIO mode by configuring direct Python execution.

Edit your Claude Desktop config (`~/.config/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "crossword-extractor": {
      "command": "python",
      "args": ["/home/jakub/crossword-read/src/mcp_server.py"],
      "env": {}
    }
  }
}
```

**Do NOT use systemd with Claude Desktop** - Claude Desktop requires STDIO transport and will launch the server process on-demand.

### Other MCP Clients

- **STDIO clients** (like Claude Desktop): Use direct Python execution, not systemd
- **HTTP/SSE clients** (web-based MCP clients): Connect to the systemd service at `http://127.0.0.1:8000/sse`

## Auto-Start on Boot

To make the service start automatically when your system boots:

```bash
# Enable the service
systemctl --user enable crossword-mcp

# Enable lingering (allows user services to run without login)
loginctl enable-linger $USER
```

To disable auto-start:

```bash
systemctl --user disable crossword-mcp
loginctl disable-linger $USER  # Optional: disables all user services
```

## Monitoring and Maintenance

### Check Service Health

```bash
# Quick health check
systemctl --user is-active crossword-mcp

# Detailed status
systemctl --user status crossword-mcp
```

### Monitor Resource Usage

```bash
# Show memory and CPU usage
systemctl --user status crossword-mcp | grep -E 'Memory|CPU'
```

### Log Rotation

Systemd journal handles log rotation automatically. To configure retention:

```bash
# Edit journal configuration (requires root)
sudo nano /etc/systemd/journald.conf

# Set retention policies, e.g.:
# SystemMaxUse=500M
# MaxRetentionSec=1month
```

## Uninstallation

To completely remove the service:

```bash
# Stop and disable the service
systemctl --user stop crossword-mcp
systemctl --user disable crossword-mcp

# Remove the service file
rm ~/.config/systemd/user/crossword-mcp.service

# Reload systemd
systemctl --user daemon-reload

# Reset failed state (if any)
systemctl --user reset-failed
```

## Advanced Configuration

### Multiple Instances

To run multiple instances on different ports or with different configurations:

1. Copy the service file:
   ```bash
   cp ~/.config/systemd/user/crossword-mcp.service \
      ~/.config/systemd/user/crossword-mcp@.service
   ```

2. Modify the service file to use instance parameters
3. Start instances: `systemctl --user start crossword-mcp@instance1`

### Environment Variables

To add custom environment variables, edit the service file:

```ini
[Service]
Environment="CUSTOM_VAR=value"
Environment="ANOTHER_VAR=value"
```

### Custom Logging

To redirect logs to a file instead of journal:

```ini
[Service]
StandardOutput=append:/path/to/logfile.log
StandardError=append:/path/to/errorlog.log
```

## Security Considerations

The service includes security hardening:
- Runs as your user (not root)
- Read-only access to system directories
- Isolated temporary directory
- No privilege escalation allowed
- Resource limits prevent runaway processes

For production deployments, consider:
- Running as a dedicated system user
- Additional SELinux or AppArmor policies
- Network isolation if not using STDIO
- Regular security updates

## Troubleshooting Common Issues

### Permission Denied Errors

```bash
# Ensure the script is executable
chmod +x /home/jakub/crossword-read/src/mcp_server.py

# Check Python permissions
ls -la $(which python)
```

### Module Not Found

```bash
# Verify dependencies are installed
python -c "import fastmcp; import cv2; import numpy"
```

### Service Won't Enable

```bash
# Check if lingering is enabled
loginctl show-user $USER | grep Linger

# Enable lingering if needed
loginctl enable-linger $USER
```

## Additional Resources

- [systemd Documentation](https://www.freedesktop.org/software/systemd/man/)
- [MCP Server Documentation](../docs/MCP_SERVER.md)
- [Main Project README](../README.md)
- [Development Guide](../CLAUDE.md)

## Support

For issues with:
- **Systemd deployment**: Check this guide and systemd logs
- **MCP server functionality**: See `docs/MCP_SERVER.md`
- **Core extraction features**: See `README.md` and `CLAUDE.md`
