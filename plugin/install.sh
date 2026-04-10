#!/usr/bin/env bash
# install.sh — Register the RepoMind MCP plugin with Claude Code (macOS / Linux)
#
# Usage:
#   bash plugin/install.sh
#
# What it does:
#   1. Installs the mcp Python package
#   2. Writes the mcpServers entry to ~/.claude/settings.json
#   3. Prints a confirmation with the registered tool list

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$PLUGIN_DIR/server.py"
SETTINGS="$HOME/.claude/settings.json"

# Prefer python3, fall back to python
PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
if [[ -z "$PYTHON" ]]; then
    echo "Error: Python not found in PATH." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------

echo ""
echo "  RepoMind — Claude Code Plugin Installer"
echo "  ────────────────────────────────────────"
echo "  Plugin : $PLUGIN_DIR"
echo "  Server : $SERVER"
echo "  Python : $PYTHON"
echo "  Target : $SETTINGS"
echo ""

# ---------------------------------------------------------------------------
# Install Python dependency
# ---------------------------------------------------------------------------

echo "› Installing dependency: mcp>=1.0.0"
"$PYTHON" -m pip install --quiet "mcp>=1.0.0"
echo "  ✓ mcp installed"
echo ""

# ---------------------------------------------------------------------------
# Ensure ~/.claude/settings.json exists
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$SETTINGS")"
if [[ ! -f "$SETTINGS" ]]; then
    echo "{}" > "$SETTINGS"
fi

# ---------------------------------------------------------------------------
# Inject mcpServers entry (idempotent)
# ---------------------------------------------------------------------------

echo "› Registering MCP server in $SETTINGS"

"$PYTHON" - <<PYEOF
import json, sys

settings_path = "$SETTINGS"
python_bin    = "$PYTHON"
server_path   = "$SERVER"

with open(settings_path) as f:
    cfg = json.load(f)

cfg.setdefault("mcpServers", {})["repomind"] = {
    "type":    "stdio",
    "command": python_bin,
    "args":    [server_path],
}

with open(settings_path, "w") as f:
    json.dump(cfg, f, indent=2)

print("  ✓ 'repomind' registered in mcpServers")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "  ✓ Installation complete."
echo ""
echo "  Restart Claude Code, then use these MCP tools:"
echo "    repomind_ask       — ask a question about the codebase"
echo "    repomind_explain   — explain a specific file"
echo "    repomind_overview  — high-level repo overview"
echo "    repomind_index     — build / update the semantic index"
echo "    repomind_doctor    — diagnose setup issues"
echo ""
echo "  Or use slash commands (if .claude/commands/ is in a project Claude Code opens):"
echo "    /repomind-ask <question>"
echo "    /repomind-explain <file>"
echo "    /repomind-overview"
echo ""
