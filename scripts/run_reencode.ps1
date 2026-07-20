$logFile = "C:\luanvan\data_eda\reports\reencode_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$venv = "C:\luanvan\data_eda\.venv-gpu\Scripts\python.exe"
$script = "C:\luanvan\data_eda\scripts\reencode_all_articles_gpu.py"

Write-Host "Starting re-encode, log: $logFile"
Write-Host "Running: $venv $script"

$env:PYTHONIOENCODING = 'utf-8'
& $venv $script 2>&1 | Tee-Object -FilePath $logFile
Write-Host "Re-encode finished, exit code: $LASTEXITCODE"
