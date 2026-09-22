param(
    [int]$WaitSeconds = 30,
    [string]$FileName = "",
    [string]$LoggingPath = ""
)

# Validate parameters
if ($WaitSeconds -le 0) {
    Write-Host "Error: WaitSeconds must be greater than 0"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($FileName)) {
    Write-Host "Error: FileName parameter is required"
    exit 1
}

# Set default logging path if not provided
if ([string]::IsNullOrWhiteSpace($LoggingPath)) {
    $LoggingPath = "$env:USERPROFILE\Downloads\ExcelKiller_Log.txt"
    Write-Host "No logging path provided. Using default: $LoggingPath"
}

# Create logging directory if it doesn't exist
$logDir = Split-Path -Parent $LoggingPath
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    Write-Host "Created logging directory: $logDir"
}

# Function to log messages
function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LoggingPath -Value $logMessage
}

Write-Log "Script started with parameters: WaitSeconds=$WaitSeconds, FileName=$FileName, LoggingPath=$LoggingPath"

# Start sleep
Write-Log "Starting sleep for $WaitSeconds seconds..."
Start-Sleep -Seconds $WaitSeconds
Write-Log "Sleep completed. Checking for non-responding Excel processes..."

try {
    # Get all non-responding Excel processes
    $hungExcelList = Get-Process EXCEL -ErrorAction SilentlyContinue | 
                     Where-Object { $_.Responding -eq $false }

    if ($hungExcelList) {
        Write-Log "Found $($hungExcelList.Count) non-responding Excel process(es)"

        # Ensure it's a list even if only one process
        if ($hungExcelList -isnot [array]) {
            $hungExcelList = @($hungExcelList)
        }

        # Loop through each non-responding process
        foreach ($process in $hungExcelList) {
            $processId = $process.Id
            $windowTitle = $process.MainWindowTitle

            Write-Log "Checking PID: $processId | Window Title: [$windowTitle]"

            # Check if window title matches the filename parameter
            if ($windowTitle -like "*$FileName*") {
                Write-Log "Match found! Window title contains '$FileName'. Killing PID: $processId"
                
                try {
                    Stop-Process -Id $processId -Force
                    Write-Log "Successfully killed process PID: $processId | FileName: $FileName"
                }
                catch {
                    Write-Log "Error killing PID $processId : $($_.Exception.Message)"
                }
            }
            else {
                Write-Log "No match. Window title [$windowTitle] does not contain '$FileName'"
            }
        }

        Write-Log "Process check completed"
        exit 0
    }
    else {
        Write-Log "No non-responding Excel processes found"
        exit 0
    }
}
catch {
    Write-Log "Error: $($_.Exception.Message)"
    exit 1
}