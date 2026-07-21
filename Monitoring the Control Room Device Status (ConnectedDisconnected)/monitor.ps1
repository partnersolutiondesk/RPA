# ==========================
# Configuration
# ==========================
$USERNAME = "your-username"
$password = "your-password"

$CHECK_INTERVAL = 30  # seconds

# Logging configuration
$logPath = "disconnected_devices.log"

# ==========================
# DeviceMonitor Class
# ==========================
class DeviceMonitor {
    [string]$baseUrl
    [string[]]$hostnames
    [string]$token
    [DateTime]$lastAuthTime  # Track when we last authenticated

    DeviceMonitor([string]$baseUrl, [string[]]$hostnames) {
        $this.baseUrl = $baseUrl.TrimEnd("/")
        $this.hostnames = $hostnames
        $this.token = $null
        $this.lastAuthTime = [DateTime]::MinValue
    }

    [void] Authenticate() {
        <#
        Step 0: Authorization API
        #>
        $authUrl = "$($this.baseUrl)/v2/authentication"

        $payload = @{
            username = $script:USERNAME
            password = $script:password
        } | ConvertTo-Json -Depth 10

        try {
            $response = Invoke-RestMethod -Uri $authUrl -Method Post -Body $payload -ContentType "application/json" -TimeoutSec 30

            # Adjust according to actual API response
            $this.token = $response.token

            if (-not $this.token) {
                throw "Token not found in auth response"
            }

            # Update last auth time
            $this.lastAuthTime = [DateTime]::Now
            
            Write-Host "Authentication successful"
            Write-Host $this.token

        } catch {
            Write-Host "Authentication failed: $_"
            throw
        }
    }

    [hashtable] BuildPayload() {
        <#
        Creates OR filter for all hostnames. If no hostnames are provided, returns all devices.
        #>

        $payload = @{
            fields = @()
            sort = @(
                @{
                    field = "hostName"
                    direction = "asc"
                }
            )
            page = @{
                offset = 0
                length = 100
            }
        }

        # Only add filter if there are hostnames to filter by
        if ($this.hostnames.Count -gt 0) {
            $operands = @()
            foreach ($hostname in $this.hostnames) {
                $operands += @{
                    field = "hostName"
                    value = $hostname
                    operator = "substring"
                }
            }
            $payload["filter"] = @{
                operator = "or"
                operands = $operands
            }
        }

        return $payload
    }

    [void] CheckDevices() {
        <#
        Step 1: Get device status
        Step 2: Log disconnected devices
        #>

        $deviceUrl = "$($this.baseUrl)/v2/devices/list"

        $headers = @{
            "X-Authorization" = $this.token
            "Content-Type" = "application/json"
        }

        $payload = $this.BuildPayload() | ConvertTo-Json -Depth 10

        try {
            try {
                $devices = Invoke-RestMethod -Uri $deviceUrl -Method Post -Headers $headers -Body $payload -ContentType "application/json" -TimeoutSec 30
            } catch {
                # Check if it's a 401 Unauthorized error (token expired)
                if ($_.Exception.Response.StatusCode -eq [System.Net.HttpStatusCode]::Unauthorized) {
                    Write-Host "Token expired, re-authenticating..."
                    $this.Authenticate()
                    
                    # Update headers with new token
                    $headers["X-Authorization"] = $this.token
                    
                    # Retry the request
                    $devices = Invoke-RestMethod -Uri $deviceUrl -Method Post -Headers $headers -Body $payload -ContentType "application/json" -TimeoutSec 30
                } else {
                    # Re-throw other errors
                    throw
                }
            }

            # Adjust path according to actual response
            $deviceList = $devices.list

            foreach ($device in $deviceList) {

                $hostname = $device.hostName
                $status = [string]$device.status.ToLower()

                Write-Host "$($hostname): $($status)"

                if ($status -ceq "disconnected") {

                    Write-Host "Device $($hostname) is DISCONNECTED"

                    $msg = "Device $($hostname) is DISCONNECTED"

                    # Logging - matches Python's logging.basicConfig format
                    $logMsg = "$(Get-Date -Format "yyyy-MM-dd HH:mm:ss") - $msg"
                    Add-Content -Path $script:logPath -Value $logMsg

                    Write-Host $msg
                }

            }

        } catch {
            Write-Host "Device check failed: $_"
        }
    }

    [void] Run() {

        $this.Authenticate()

        while ($true) {
            try {
                # Proactively re-authenticate every 15 minutes (900 seconds)
                if ($this.lastAuthTime -ne [DateTime]::MinValue -and ([DateTime]::Now - $this.lastAuthTime).TotalSeconds -gt 900) {
                    Write-Host "15 minutes passed, proactively re-authenticating..."
                    $this.Authenticate()
                }
                
                Write-Host "`n[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Checking devices..."

                $this.CheckDevices()

            } catch {
                Write-Host "Error: $_"
            }

            Start-Sleep -Seconds $script:CHECK_INTERVAL
        }
    }
}

# ==========================
# Main Script
# ==========================
$baseUrl = "YOUR_BASE_URL_HERE"

$hostnames = @() #acts as filter eg;"n9k", "ywz"

$monitor = [DeviceMonitor]::new($baseUrl, $hostnames)

$monitor.Run()
