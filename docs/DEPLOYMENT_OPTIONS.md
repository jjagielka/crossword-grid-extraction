# Deployment Options for Crossword MCP Server

This document explains the different ways to run the Crossword MCP server.

## Quick Decision Guide

**Choose your deployment based on your use case:**

| Use Case | Mode | Deployment Method | When to Use |
|----------|------|-------------------|-------------|
| Claude Desktop (Linux/Mac/Windows) | STDIO | Direct execution | Default for desktop clients |
| Always-on server for web clients | HTTP/SSE | Systemd service | Remote access, web apps |
| Development/Testing | STDIO or HTTP | Direct execution | Quick testing |
| Docker deployment | HTTP/SSE | Docker + systemd | Containerized environments |

## Deployment Mode Details

### 1. STDIO Mode (Default)

**What it is:**
- Server communicates via standard input/output (pipes)
- Client launches the server process on-demand
- Server exits when client disconnects

**When to use:**
- Claude Desktop on any platform
- Desktop MCP clients
- Command-line MCP tools
- On-demand processing

**How to run:**
```bash
python src/mcp_server.py
```

**Configuration example (Claude Desktop on Linux):**
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

**Advantages:**
- Simple, no server management needed
- Client handles lifecycle
- No port conflicts
- Low resource usage when idle

**Disadvantages:**
- Can't be shared between multiple clients
- Can't be accessed remotely
- Not suitable for systemd services

---

### 2. HTTP/SSE Mode

**What it is:**
- Server runs as persistent HTTP service
- Listens on port 8000 (default)
- Uses Server-Sent Events (SSE) for MCP communication
- Includes uvicorn internally (no separate ASGI server needed)

**When to use:**
- Multiple clients need access
- Remote access required
- Always-on deployment
- Web-based MCP clients
- Systemd service deployment

**How to run:**
```bash
# Direct execution
python src/mcp_server.py --http

# As systemd service (see SYSTEMD_DEPLOYMENT.md)
systemctl --user start crossword-mcp
```

**Server URL:**
```
http://127.0.0.1:8000/sse
```

**Advantages:**
- Can serve multiple clients
- Always available
- Can be accessed remotely (with proper network config)
- Managed by systemd (auto-restart, logging, etc.)

**Disadvantages:**
- Uses resources even when idle
- Requires port management
- More complex setup

---

## Platform-Specific Recommendations

### Linux

**For Claude Desktop:**
- Use STDIO mode with direct Python execution
- Configure in `~/.config/Claude/claude_desktop_config.json`

**For server deployment:**
- Use HTTP/SSE mode with systemd
- See `docs/SYSTEMD_DEPLOYMENT.md`
- Enables auto-start on boot with `loginctl enable-linger`

### macOS

**For Claude Desktop:**
- Use STDIO mode with direct Python execution
- Configure in `~/Library/Application Support/Claude/claude_desktop_config.json`

**For server deployment:**
- Use HTTP/SSE mode with launchd (not covered in this guide)
- Or run manually in a tmux/screen session

### Windows

**For Claude Desktop:**
- Use STDIO mode with direct Python execution
- Configure in `%APPDATA%\Claude\claude_desktop_config.json`
- Use Python from PATH or specify full path if needed

**For server deployment:**
- Use HTTP/SSE mode with NSSM (Non-Sucking Service Manager)
- Or use Windows Task Scheduler
- Systemd not available on Windows

---

## Common Scenarios

### Scenario 1: Local Development with Claude Desktop

**Best choice:** STDIO mode

**Setup:**
```json
{
  "mcpServers": {
    "crossword-extractor": {
      "command": "python",
      "args": ["/home/jakub/crossword-read/src/mcp_server.py"]
    }
  }
}
```

**Why:** Simple, automatic lifecycle management, no server needed.

---

### Scenario 2: Always-On Linux Server

**Best choice:** HTTP/SSE mode with systemd

**Setup:**
```bash
# Install systemd service
./systemd/install.sh

# Start and enable
systemctl --user start crossword-mcp
systemctl --user enable crossword-mcp
loginctl enable-linger $USER
```

**Why:** Survives reboots, automatic restart, centralized logging, resource limits.

---

### Scenario 3: Multiple Users on Same Machine

**Best choice:** HTTP/SSE mode with systemd

**Setup:**
- Run systemd service in HTTP/SSE mode
- Each user's MCP client connects to `http://localhost:8000/sse`

**Why:** Single server instance serves all users, efficient resource usage.

---

### Scenario 4: Remote Access

**Best choice:** HTTP/SSE mode with reverse proxy

**Setup:**
1. Run systemd service in HTTP/SSE mode
2. Configure nginx/Apache reverse proxy:
   ```nginx
   location /crossword-mcp/ {
       proxy_pass http://127.0.0.1:8000/;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
   }
   ```
3. Add authentication and HTTPS

**Why:** Secure remote access, can add auth, SSL, rate limiting.

---

## Testing Your Deployment

### Test STDIO Mode

```bash
# Should show MCP banner and wait for input
python src/mcp_server.py

# Press Ctrl+C to exit
```

### Test HTTP/SSE Mode

```bash
# Start server
python src/mcp_server.py --http

# In another terminal, test endpoints
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/sse
```

### Test Systemd Service

```bash
# Check if running
systemctl --user status crossword-mcp

# View logs
journalctl --user -u crossword-mcp -f

# Test endpoint
curl http://127.0.0.1:8000/health
```

---

## Troubleshooting

### "Port 8000 already in use"

**Solution:**
```bash
# Find what's using the port
lsof -i :8000

# Or change the port (modify mcp_server.py to add --port argument)
```

### "systemd service fails to start"

**Solution:**
```bash
# Check logs
journalctl --user -u crossword-mcp -n 50

# Verify Python and dependencies
python -c "import fastmcp; import cv2"

# Test manually
python src/mcp_server.py --http
```

### "Claude Desktop can't connect"

**Solution:**
- Verify you're using STDIO mode (not --http)
- Check Python is in PATH or specify full path
- Ensure all dependencies are installed
- Check Claude Desktop logs

---

## Performance Considerations

### STDIO Mode
- **Memory:** ~100-300MB per instance
- **CPU:** Only when processing
- **Startup time:** 1-2 seconds
- **Ideal for:** On-demand processing

### HTTP/SSE Mode
- **Memory:** ~100-300MB persistent
- **CPU:** Minimal when idle, spikes during processing
- **Startup time:** 2-3 seconds
- **Idle overhead:** ~5-10MB RAM, <1% CPU
- **Ideal for:** Always-on services

---

## Security Considerations

### STDIO Mode
- Runs with user permissions
- No network exposure
- Process isolation per client
- **Security level:** High

### HTTP/SSE Mode (local)
- Listens on 127.0.0.1 only (not exposed to network)
- Runs with user permissions
- Systemd security hardening applied
- **Security level:** Medium-High

### HTTP/SSE Mode (remote)
- **Requires additional security:**
  - HTTPS/TLS encryption
  - Authentication (API keys, OAuth, etc.)
  - Rate limiting
  - Firewall rules
  - Regular security updates
- **Security level:** Depends on configuration

---

## Additional Resources

- **Systemd deployment guide:** `docs/SYSTEMD_DEPLOYMENT.md`
- **MCP server documentation:** `docs/MCP_SERVER.md`
- **Project README:** `README.md`
- **FastMCP documentation:** https://gofastmcp.com
- **MCP specification:** https://modelcontextprotocol.io
