$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = 8061
$url = "http://127.0.0.1:$port/"
$viewerScript = Join-Path $root "viewer_server.py"
$tag = "[open_viewer]"

function Log([string]$Message) {
    Write-Host "$tag $Message"
}

function Stop-ExistingServer([int]$PortNumber) {
    Log "Cleaning old listeners on port $PortNumber..."
    try {
        $pids = Get-NetTCPConnection -LocalPort $PortNumber -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($pid in $pids) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction Stop
                Log "Stopped PID=$pid"
            } catch {
                Log "Failed to stop PID=$pid : $($_.Exception.Message)"
            }
        }
    } catch {
        Log "No old process on port $PortNumber or clean-up not needed."
    }
}

function Wait-ViewerServer([string]$Url, [int]$RetryCount = 80, [int]$DelayMs = 250) {
    for ($i = 0; $i -lt $RetryCount; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Milliseconds $DelayMs
    }
    return $false
}

if (-not (Test-Path -LiteralPath $viewerScript -PathType Leaf)) {
    throw "Viewer script not found: $viewerScript"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $pythonCommand) {
    throw "Python not found in PATH."
}

Stop-ExistingServer -PortNumber $port
Log "Using python: $($pythonCommand.Source)"
Log "Start viewer: $viewerScript"

$serverProc = Start-Process -FilePath $pythonCommand.Source -ArgumentList @($viewerScript) -WorkingDirectory $root -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 1
if ($serverProc.HasExited) {
    throw "Viewer process exited early. ExitCode=$($serverProc.ExitCode)."
}

if (-not (Wait-ViewerServer -Url $url)) {
    throw "Server not ready after timeout: $url"
}

Log "Server is ready. Open url: $url"
Start-Process $url
