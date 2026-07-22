<#
.SYNOPSIS
    Watchdog — monitors agents, restarts failed ones, generates consolidated report.
    Run as a scheduled task or standalone loop.
#>

param(
    [int]$IntervalMinutes = 30,
    [switch]$Loop
)

$ProjectRoot = "C:\luanvan\data_eda"
$PidsFile = Join-Path $ProjectRoot "results\research_agent\agent_pids.json"
$LogFile = Join-Path $ProjectRoot "results\research_agent\logs\watchdog.log"

function Write-Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts [WATCHDOG] $Message" | Out-File -FilePath $LogFile -Append -Encoding utf8
    Write-Host "$ts [WATCHDOG] $Message"
}

function Test-AgentAlive {
    param([string]$Name, [int]$Pid)
    $proc = Get-Process -Id $Pid -ErrorAction SilentlyContinue
    return ($null -ne $proc -and (-not $proc.HasExited))
}

function Invoke-AgentCheck {
    Write-Log "Checking agent health..."
    if (-not (Test-Path $PidsFile)) {
        Write-Log "No PIDs file — agents not running. Attempting launch..."
        & (Join-Path $ProjectRoot "scripts\launch_agents.ps1") -Action start
        return
    }
    
    $pids = Get-Content $PidsFile -Raw | ConvertFrom-Json
    $anyFailed = $false
    foreach ($entry in $pids.PSObject.Properties) {
        $name = $entry.Name
        $pid = $entry.Value
        if (-not (Test-AgentAlive -Name $name -Pid $pid)) {
            Write-Log "  $name (PID $pid) is DEAD. Restarting..."
            $anyFailed = $true
        } else {
            Write-Log "  $name (PID $pid) OK"
        }
    }
    
    if ($anyFailed) {
        Write-Log "Restarting all agents..."
        & (Join-Path $ProjectRoot "scripts\launch_agents.ps1") -Action start
    }
    
    # Generate consolidated report
    Write-Log "Generating consolidated report..."
    try {
        $env:PYTHONIOENCODING='utf-8'
        $result = & uv run python -m src.research_agent.agent_consolidated 2>&1
        Write-Log "Report done"
    } catch {
        Write-Log "Report failed: $_"
    }
}

# Single shot
if (-not $Loop) {
    Invoke-AgentCheck
    exit
}

# Loop mode
Write-Log "Watchdog started (interval=${IntervalMinutes}m)"
while ($true) {
    Invoke-AgentCheck
    Write-Log "Sleeping $IntervalMinutes minutes..."
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
