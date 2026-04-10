# install.ps1 — Register the RepoMind MCP plugin with Claude Code (Windows)
#
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File plugin\install.ps1
#
# What it does:
#   1. Installs the mcp Python package
#   2. Writes the mcpServers entry to %USERPROFILE%\.claude\settings.json
#   3. Prints a confirmation with the registered tool list

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

$PluginDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Server    = Join-Path $PluginDir "server.py"
$Settings  = Join-Path $env:USERPROFILE ".claude\settings.json"

$Python = $(
    try { (Get-Command python3 -ErrorAction Stop).Source }
    catch {
        try { (Get-Command python -ErrorAction Stop).Source }
        catch { $null }
    }
)

if (-not $Python) {
    Write-Error "Python not found in PATH."
    exit 1
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  RepoMind - Claude Code Plugin Installer"
Write-Host "  ----------------------------------------"
Write-Host "  Plugin : $PluginDir"
Write-Host "  Server : $Server"
Write-Host "  Python : $Python"
Write-Host "  Target : $Settings"
Write-Host ""

# ---------------------------------------------------------------------------
# Install Python dependency
# ---------------------------------------------------------------------------

Write-Host "> Installing dependency: mcp>=1.0.0"
& $Python -m pip install --quiet "mcp>=1.0.0"
Write-Host "  OK  mcp installed"
Write-Host ""

# ---------------------------------------------------------------------------
# Ensure %USERPROFILE%\.claude\settings.json exists
# ---------------------------------------------------------------------------

$SettingsDir = Split-Path $Settings
if (-not (Test-Path $SettingsDir)) {
    New-Item -ItemType Directory -Path $SettingsDir | Out-Null
}
if (-not (Test-Path $Settings)) {
    '{}' | Set-Content -Path $Settings -Encoding UTF8
}

# ---------------------------------------------------------------------------
# Inject mcpServers entry (idempotent)
# ---------------------------------------------------------------------------

Write-Host "> Registering MCP server in $Settings"

$Script = @"
import json

settings_path = r'$Settings'
python_bin    = r'$Python'
server_path   = r'$Server'

with open(settings_path, encoding='utf-8') as f:
    cfg = json.load(f)

cfg.setdefault('mcpServers', {})['repomind'] = {
    'type':    'stdio',
    'command': python_bin,
    'args':    [server_path],
}

with open(settings_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)

print("  OK  'repomind' registered in mcpServers")
"@

& $Python -c $Script

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  OK  Installation complete."
Write-Host ""
Write-Host "  Restart Claude Code, then use these MCP tools:"
Write-Host "    repomind_ask       - ask a question about the codebase"
Write-Host "    repomind_explain   - explain a specific file"
Write-Host "    repomind_overview  - high-level repo overview"
Write-Host "    repomind_index     - build / update the semantic index"
Write-Host "    repomind_doctor    - diagnose setup issues"
Write-Host ""
