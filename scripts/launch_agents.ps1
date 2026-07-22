<#
.SYNOPSIS
    Launch all 5 research agents as independent background processes.
    Agents survive terminal disconnection.
.DESCRIPTION
    Uses pythonw.exe (windowless Python) for true background operation.
    PIDs tracked in results/research_agent/agent_pids.json.
    Use stop_agents.ps1 to kill all agents.
.PARAMETER Action
    "start" (default) — launch all agents
    "stop" — kill all running agents
    "status" — check agent status
    "restart" — stop then start
#>

param(
    [ValidateSet("start", "stop", "status", "restart")]
    [string]$Action = "start"
)

$ProjectRoot = "C:\luanvan\data_eda"
$PidsFile = Join-Path $ProjectRoot "results\research_agent\agent_pids.json"
$LogsDir = Join-Path $ProjectRoot "results\research_agent\logs"
$AgentDir = Join-Path $ProjectRoot "src\research_agent"

# Find pythonw.exe (use venv first)
$VenvPythonw = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
if (Test-Path $VenvPythonw) {
    $Pythonw = $VenvPythonw
} else {
    $Pythonw = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
    if (-not $Pythonw) {
        $Pythonw = (Get-Command python -ErrorAction SilentlyContinue).Source -replace 'python\.exe$', 'pythonw.exe'
    }
}
if (-not (Test-Path $Pythonw)) {
    Write-Error "pythonw.exe not found"
    exit 1
}
Write-Host "Using pythonw: $Pythonw"

$Agents = @(
    @{name="sentiment"; module="src.research_agent.agent_sentiment"; log="agent_sentiment.log"}
    @{name="model"; module="src.research_agent.agent_model"; log="agent_model.log"}
    @{name="feature"; module="src.research_agent.agent_feature"; log="agent_feature.log"}
    @{name="horizon"; module="src.research_agent.agent_horizon"; log="agent_horizon.log"}
    @{name="ticker"; module="src.research_agent.agent_ticker"; log="agent_ticker.log"}
)

function Stop-Agents {
    Write-Host "Stopping agents..."
    if (Test-Path $PidsFile) {
        $pids = Get-Content $PidsFile -Raw | ConvertFrom-Json
        foreach ($agent in $Agents) {
            $name = $agent.name
            if ($pids.$name -and (Get-Process -Id $pids.$name -ErrorAction SilentlyContinue)) {
                Write-Host "  Killing $name (PID $($pids.$name))..."
                Stop-Process -Id $pids.$name -Force -ErrorAction SilentlyContinue
            }
        }
        Remove-Item $PidsFile -Force -ErrorAction SilentlyContinue
    }
    # Kill any stray pythonw research_agent processes
    Get-Process -Name "pythonw" -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match "research_agent"
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "All agents stopped."
}

function Status-Agents {
    Write-Host "`n=== Agent Status ==="
    if (Test-Path $PidsFile) {
        $pids = Get-Content $PidsFile -Raw | ConvertFrom-Json
        foreach ($agent in $Agents) {
            $name = $agent.name
            $pid = $pids.$name
            if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) {
                $logFile = Join-Path $LogsDir $agent.log
                $lastLine = if (Test-Path $logFile) { (Get-Content $logFile -Tail 1 -ErrorAction SilentlyContinue) } else { "No log" }
                Write-Host "  [RUNNING] $name (PID $pid) -> $lastLine" -ForegroundColor Green
            } else {
                Write-Host "  [STOPPED] $name" -ForegroundColor Red
            }
        }
    } else {
        foreach ($agent in $Agents) {
            Write-Host "  [STOPPED] $($agent.name)" -ForegroundColor Red
        }
    }
}

function Start-Agents {
    Write-Host "Starting agents..."
    
    # Kill existing first
    Stop-Agents
    
    # Create log dir
    if (-not (Test-Path $LogsDir)) { New-Item -ItemType Directory -Path $LogsDir -Force }
    
    $pidTable = @{}
    foreach ($agent in $Agents) {
        $logFile = Join-Path $LogsDir $agent.log
        $name = $agent.name
        Write-Host "  Launching $name..."
        
        try {
            $proc = Start-Process -FilePath $Pythonw -ArgumentList "-m", $agent.module `
                -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput "$logFile.stdout" -RedirectStandardError "$logFile.stderr"
            Start-Sleep -Milliseconds 500
            if (-not $proc.HasExited) {
                $pidTable[$name] = $proc.Id
                Write-Host "    Started (PID $($proc.Id))" -ForegroundColor Green
            } else {
                Write-Host "    FAILED to start (exited immediately)" -ForegroundColor Red
            }
        } catch {
            Write-Host "    ERROR: $_" -ForegroundColor Red
        }
    }
    
    # Save PIDs
    $pidTable | ConvertTo-Json | Set-Content -Path $PidsFile -Force
    Write-Host "`nPIDs saved to $PidsFile"
}

switch ($Action) {
    "start" { Start-Agents; Status-Agents }
    "stop" { Stop-Agents }
    "status" { Status-Agents }
    "restart" { Stop-Agents; Start-Agents; Status-Agents }
}
