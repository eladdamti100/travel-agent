$logDir = ".\logs"
$logFile = "$logDir\travel_agent.log"

if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

if (!(Test-Path $logFile)) {
    New-Item -ItemType File -Path $logFile | Out-Null
}

Write-Host "Watching logs: $logFile" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray

Get-Content $logFile -Wait -Tail 80