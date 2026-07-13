$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8061

Write-Host "Starting PLY viewer at http://127.0.0.1:$port/"
Write-Host "Root: $root"
Write-Host "Press Ctrl+C to stop the server."

python (Join-Path $root "viewer_server.py")
