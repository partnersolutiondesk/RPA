<#
.SYNOPSIS
    Automation Anywhere 360 (v30) -- Role & Directory Permission Report

.DESCRIPTION
    APIs used:
      1. POST /v2/authentication                                          -> bearer token
      2. POST /v2/usermanagement/roles/list                              -> paginated roles
      3. GET  /v1/repository/role/{roleId}/directories                   -> top-level directories for a role
      4. GET  /v1/repository/role/{roleId}/directories/{dirId}/directories -> subdirectories (recursive)

    Output: Role_Directory_Report.xlsx  (3 sheets)
      Sheet 1 -- Roles          : one row per role (id, name, description, counts, created/modified)
      Sheet 2 -- Directories    : one row per role+directory (flattened tree, all depths)
      Sheet 3 -- Scan Issues    : any API failures recorded during the run

    Requirements:
        Install-Module ImportExcel -Scope CurrentUser

    Usage:
        .\Role_Directory_Permission_Scanner.ps1
        -- or override via environment variables: AA_CR_URL, AA_USERNAME, AA_PASSWORD
#>

#Requires -Version 5.1

[CmdletBinding()]
param()

try   { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
$CR_URL   = if ($env:AA_CR_URL)   { $env:AA_CR_URL }   else { "---CR--URL--" }
$USERNAME = if ($env:AA_USERNAME) { $env:AA_USERNAME } else { "--USER-NAME--" }
$PASSWORD = if ($env:AA_PASSWORD) { $env:AA_PASSWORD } else { "--PASSWORD--" }


$CR_URL   = $CR_URL.Trim().TrimEnd('/')
$USERNAME = $USERNAME.Trim()
$PASSWORD = $PASSWORD.Trim()

$OUTPUT_FILE = "Role_Directory_Report.xlsx"
$LOG_FILE    = "Role_Directory_Report.log"
$PAGE_SIZE   = 100      # roles per page (max 100)
$DELAY_MS    = 100      # milliseconds between API calls
$VERIFY_SSL  = $true

# Set $true on first run to dump raw API responses for inspection
$DEBUG_DUMP  = $false

# ═══════════════════════════════════════════════════════════════════════════════
#  RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
$MAX_RETRIES        = 1
$RETRY_DELAY_SEC    = 2
$RETRY_BACKOFF_MULT = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
#  RE-AUTH CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════
$MAX_REAUTH_FAILURES = 5
$REAUTH_PAUSE_SEC    = 300

# ═══════════════════════════════════════════════════════════════════════════════
#  PROACTIVE TOKEN REFRESH
# ═══════════════════════════════════════════════════════════════════════════════
$TOKEN_LIFETIME_SEC = 900   # re-auth every 15 min (token expires ~20 min)

# ─────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────
$script:AuthHeaders               = @{}
$script:TokenFetchedAt            = [DateTime]::MinValue
$script:ConsecutiveReAuthFailures = 0
$script:FailedApiCalls            = [System.Collections.ArrayList]@()
$script:RoleCounter               = 0
$script:DirCounter                = 0

# ── Incremental Excel state ───────────────────────────────────────────────────
$script:Excel_RoleRow  = 2    # next write row on Roles sheet
$script:Excel_DirRow   = 2    # next write row on Directories sheet

# Fixed permission field order (API field name -> UI column label)
$script:PermFieldMap = [ordered]@{
    'canRun'         = 'Run and Schedule'
    'canUpload'      = 'Check In'
    'canDownload'    = 'Check Out'
    'canViewContent' = 'View Content'
    'canClone'       = 'Clone'
    'canDelete'      = 'Delete from Public'
    'canManage'      = 'Manage'
}


# ─────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("INFO","WARN","ERROR","DEBUG")]
        [string]$Level = "INFO"
    )
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "$timestamp [$Level] $Message"
    switch ($Level) {
        "ERROR" { Write-Host $line -ForegroundColor Red    }
        "WARN"  { Write-Host $line -ForegroundColor Yellow }
        "DEBUG" { if ($DebugPreference -ne 'SilentlyContinue') { Write-Host $line -ForegroundColor Gray } }
        default { Write-Host $line -ForegroundColor Cyan   }
    }
    try {
        $logPath = Join-Path (Get-Location).Path $LOG_FILE
        $line | Out-File -FilePath $logPath -Append -Encoding UTF8
    } catch {}
}

function Write-LogBanner {
    param([string]$Message)
    Write-Host $Message
    try {
        $logPath = Join-Path (Get-Location).Path $LOG_FILE
        $Message | Out-File -FilePath $logPath -Append -Encoding UTF8
    } catch {}
}


# ─────────────────────────────────────────────────────
#  SSL BYPASS  (when VERIFY_SSL = $false)
# ─────────────────────────────────────────────────────
if (-not $VERIFY_SSL) {
    if ($PSVersionTable.PSVersion.Major -lt 6) {
        if (-not ([System.Management.Automation.PSTypeName]'TrustAllCertsPolicy').Type) {
            Add-Type @"
using System.Net;
using System.Security.Cryptography.X509Certificates;
public class TrustAllCertsPolicy : ICertificatePolicy {
    public bool CheckValidationResult(ServicePoint srvPoint,
        X509Certificate certificate, WebRequest request,
        int certificateProblem) { return true; }
}
"@
        }
        [System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy
        [System.Net.ServicePointManager]::SecurityProtocol  = [System.Net.SecurityProtocolType]::Tls12
    }
}


# ─────────────────────────────────────────────────────
#  HELPER: build Invoke-RestMethod param splat
# ─────────────────────────────────────────────────────
function Get-IrmParams {
    param(
        [string]$Uri,
        [string]$Method,
        [string]$Body    = $null,
        [hashtable]$Headers
    )
    $p = @{ Uri = $Uri; Method = $Method; ContentType = "application/json" }
    if ($Body)    { $p.Body    = $Body    }
    if ($Headers) { $p.Headers = $Headers }
    if (-not $VERIFY_SSL -and $PSVersionTable.PSVersion.Major -ge 6) {
        $p.SkipCertificateCheck = $true
    }
    return $p
}


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════
function Invoke-AAAuthenticate {
    $authMaxRetries = 3
    for ($attempt = 1; $attempt -le $authMaxRetries; $attempt++) {
        Write-Log "Authenticating to $CR_URL ... (attempt $attempt/$authMaxRetries)"
        $body   = @{ username = $USERNAME; password = $PASSWORD } | ConvertTo-Json
        $params = Get-IrmParams -Uri "$CR_URL/v2/authentication" -Method "POST" -Body $body

        try {
            $resp  = Invoke-RestMethod @params -ErrorAction Stop
            $token = $resp.token
            if (-not $token) { throw "No token in authentication response" }
            $script:AuthHeaders = @{
                "X-Authorization" = $token
                "Content-Type"    = "application/json"
            }
            $script:TokenFetchedAt = [DateTime]::Now
            Write-Log "Authenticated successfully."
            return
        } catch {
            $statusCode = 0
            if ($_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
            }
            if ($statusCode -eq 401) {
                throw "Authentication failed (401 Unauthorized): Invalid credentials for '$USERNAME'."
            }
            if ($attempt -ge $authMaxRetries) {
                throw "Authentication failed after $authMaxRetries attempts (HTTP $statusCode): $($_.Exception.Message)"
            }
            $delaySec = $attempt * 5
            Write-Log "Auth attempt $attempt failed (HTTP $statusCode). Retrying in ${delaySec}s..." -Level WARN
            Start-Sleep -Seconds $delaySec
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CENTRALISED RETRY WRAPPER  (with proactive token refresh + re-auth on 401)
# ═══════════════════════════════════════════════════════════════════════════════
function Invoke-ApiWithRetry {
    param(
        [string]$Uri,
        [string]$Method,
        [string]$Body        = $null,
        [string]$Description = "API call"
    )

    # Proactive token refresh before the call
    if (([DateTime]::Now - $script:TokenFetchedAt).TotalSeconds -ge $TOKEN_LIFETIME_SEC) {
        Write-Log "[ProactiveAuth] Token age >= ${TOKEN_LIFETIME_SEC}s. Refreshing before $Description..." -Level INFO
        try {
            Invoke-AAAuthenticate
        } catch {
            Write-Log "[ProactiveAuth] Pre-call re-auth failed (will retry on 401 if needed): $($_.Exception.Message)" -Level WARN
        }
    }

    $totalAttempts = $MAX_RETRIES + 1

    for ($attempt = 1; $attempt -le $totalAttempts; $attempt++) {
        $irmParams = Get-IrmParams -Uri $Uri -Method $Method `
                                   -Body $Body -Headers $script:AuthHeaders
        try {
            return Invoke-RestMethod @irmParams -ErrorAction Stop

        } catch {
            $statusCode = 0
            if ($_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
            }
            $isLast = ($attempt -ge $totalAttempts)

            # 401: token expired -> re-authenticate
            if ($statusCode -eq 401) {
                if ($isLast) {
                    Write-Log "[Retry] $Description -- 401 FINAL FAILURE after $MAX_RETRIES retries." -Level ERROR
                    throw
                }
                Write-Log "[Retry] $Description -- 401 token expired (attempt $attempt/$totalAttempts). Re-authenticating..." -Level WARN
                try {
                    Invoke-AAAuthenticate
                    $script:ConsecutiveReAuthFailures = 0
                    Write-Log "[Retry] Token refreshed. Retrying $Description..." -Level INFO
                } catch {
                    $script:ConsecutiveReAuthFailures++
                    Write-Log ("[CircuitBreaker] Re-auth failed ($($script:ConsecutiveReAuthFailures)/$MAX_REAUTH_FAILURES): $($_.Exception.Message)") -Level ERROR
                    if ($script:ConsecutiveReAuthFailures -ge $MAX_REAUTH_FAILURES) {
                        throw ("CIRCUIT BREAKER OPEN: $MAX_REAUTH_FAILURES consecutive re-auth failures. Aborting.")
                    }
                    if ($script:ConsecutiveReAuthFailures -ge 3) {
                        Write-Log "[CircuitBreaker] Pausing ${REAUTH_PAUSE_SEC}s..." -Level WARN
                        Start-Sleep -Seconds $REAUTH_PAUSE_SEC
                    }
                }

            # 429 / 5xx / network timeout
            } elseif ($statusCode -eq 429 -or
                      ($statusCode -ge 500 -and $statusCode -lt 600) -or
                      $statusCode -eq 0) {
                if ($isLast) {
                    Write-Log "[Retry] $Description -- HTTP $statusCode FINAL FAILURE after $MAX_RETRIES retries." -Level ERROR
                    throw
                }
                $delaySec = [int]([Math]::Round($RETRY_DELAY_SEC * [Math]::Pow($RETRY_BACKOFF_MULT, $attempt - 1)))
                Write-Log "[Retry] $Description -- HTTP $statusCode transient error. Backoff ${delaySec}s (attempt $attempt/$totalAttempts)..." -Level WARN
                Start-Sleep -Seconds $delaySec

            # 400 / 403 / 404 and anything else: non-retryable
            } else {
                throw
            }
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════
function Assert-Config {
    $errors = [System.Collections.ArrayList]@()
    if ([string]::IsNullOrWhiteSpace($CR_URL) -or -not $CR_URL.StartsWith("https://")) {
        [void]$errors.Add("CR_URL must start with https:// (got: '$CR_URL'). Set AA_CR_URL environment variable.")
    }
    if ([string]::IsNullOrWhiteSpace($USERNAME)) {
        [void]$errors.Add("USERNAME is not configured. Set AA_USERNAME environment variable.")
    }
    if ([string]::IsNullOrWhiteSpace($PASSWORD)) {
        [void]$errors.Add("PASSWORD is not configured. Set AA_PASSWORD environment variable.")
    }
    if ($errors.Count -gt 0) {
        foreach ($e in $errors) { Write-Log "  - $e" -Level ERROR }
        throw "Configuration validation failed. Correct the above errors and re-run."
    }
    Write-Log "Config OK: URL=$CR_URL  User=$USERNAME"
}

function Test-OutputFileLock {
    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
    if (Test-Path $outPath) {
        try {
            $stream = [System.IO.File]::Open($outPath, [System.IO.FileMode]::Open,
                      [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            $stream.Close(); $stream.Dispose()
            Write-Log "Output file exists and is writable: $outPath"
        } catch {
            throw "Output file is locked: '$outPath'. Close it in Excel first. Error: $($_.Exception.Message)"
        }
    } else {
        try {
            $stream = [System.IO.File]::Open($outPath, [System.IO.FileMode]::Create,
                      [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
            $stream.Close(); $stream.Dispose()
            Remove-Item $outPath -Force -ErrorAction SilentlyContinue
            Write-Log "Output path is writable: $outPath"
        } catch {
            throw "Cannot create output file at '$outPath'. Check permissions. Error: $($_.Exception.Message)"
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  API: GET ALL ROLES  (paginated)
# ═══════════════════════════════════════════════════════════════════════════════
function Get-AllRoles {
    $url    = "$CR_URL/v2/usermanagement/roles/list"
    $roles  = [System.Collections.ArrayList]@()
    $offset = 0

    do {
        $body = @{
            fields = @()
            filter = $null
            sort   = @( @{ field = "name"; direction = "asc" } )
            page   = @{ offset = $offset; length = $PAGE_SIZE }
        } | ConvertTo-Json -Depth 5

        try {
            $data = Invoke-ApiWithRetry -Uri $url -Method "POST" -Body $body `
                                        -Description "ListRoles offset=$offset"
        } catch {
            Write-Log "ERROR fetching roles at offset $offset : $($_.Exception.Message)" -Level ERROR
            break
        }

        $batch = if ($data.list) { @($data.list) } else { @() }
        foreach ($r in $batch) { [void]$roles.Add($r) }

        $total  = if ($data.page -and $null -ne $data.page.totalFilter) {
                      [int]$data.page.totalFilter
                  } elseif ($data.page -and $null -ne $data.page.total) {
                      [int]$data.page.total
                  } else { 0 }

        Write-Log "  Fetched $($roles.Count) / $total roles..."
        $offset += $batch.Count
        if ($batch.Count -gt 0 -and $offset -lt $total) {
            Start-Sleep -Milliseconds $DELAY_MS
        }
    } while ($offset -lt $total -and $batch.Count -gt 0)

    return ,$roles
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS: extract permission fields from a directory object
# ═══════════════════════════════════════════════════════════════════════════════
function Get-PermissionString {
    <#
    .SYNOPSIS
        Extracts permission fields from a directory object returned by the API.
        Handles both a nested "permission" object and top-level boolean fields.
        Returns a human-readable string: "view=Yes; run=Yes; edit=No; ..."
    #>
    param($DirObj)

    $parts = [System.Collections.ArrayList]@()

    # Strategy 1: nested "permission" or "permissions" object with boolean properties
    $permObj = if ($DirObj.permission)   { $DirObj.permission }  `
               elseif ($DirObj.permissions) { $DirObj.permissions } `
               else { $null }

    if ($permObj) {
        if ($permObj -is [PSCustomObject]) {
            foreach ($prop in $permObj.PSObject.Properties) {
                $val = $prop.Value
                if ($val -is [bool]) {
                    [void]$parts.Add("$($prop.Name)=$( if ($val) { 'Yes' } else { 'No' } )")
                } elseif ($null -ne $val -and [string]$val -ne '') {
                    [void]$parts.Add("$($prop.Name)=$val")
                }
            }
        } elseif ($permObj -is [array] -or $permObj -is [System.Collections.IEnumerable]) {
            # Array of permission objects e.g. [{action:"run"},{action:"view"}]
            foreach ($p in $permObj) {
                $action = if ($p.action) { [string]$p.action } `
                          elseif ($p.name) { [string]$p.name } `
                          elseif ($p.type) { [string]$p.type } `
                          else { [string]$p }
                if ($action) { [void]$parts.Add($action) }
            }
        } else {
            [void]$parts.Add([string]$permObj)
        }
    }

    # Strategy 2: top-level boolean permission fields on the directory object itself
    if ($parts.Count -eq 0) {
        $knownPermFields = @('canView','canRun','canEdit','canDelete','canCreate',
                             'canUpload','canDownload','canClone','canExport',
                             'view','run','edit','delete','create','upload',
                             'download','execute','manage','read','write')
        foreach ($f in $knownPermFields) {
            if ($null -ne $DirObj.$f) {
                $val = $DirObj.$f
                if ($val -is [bool]) {
                    [void]$parts.Add("$f=$( if ($val) { 'Yes' } else { 'No' } )")
                }
            }
        }
    }

    if ($parts.Count -eq 0) { return "N/A" }
    return ($parts -join "; ")
}

function Get-PermissionColumns {
    <#
    .SYNOPSIS
        Returns an ordered hashtable of the 7 known permission fields from a directory object.
        Values are $true / $false (booleans).
        Uses case-insensitive property lookup so canViewcontent matches canViewContent etc.
    #>
    param($DirObj)

    $cols = [ordered]@{}
    # Build a case-insensitive property map of the dir object once
    $propMap = @{}
    if ($DirObj -and $DirObj.PSObject) {
        foreach ($p in $DirObj.PSObject.Properties) {
            $propMap[$p.Name.ToLower()] = $p.Value
        }
    }

    foreach ($apiField in $script:PermFieldMap.Keys) {
        $val = $propMap[$apiField.ToLower()]
        $cols[$apiField] = ($null -ne $val -and [bool]$val -eq $true)
    }
    return $cols
}


# ═══════════════════════════════════════════════════════════════════════════════
#  API: GET DIRECTORIES FOR A ROLE  (recursive)
# ═══════════════════════════════════════════════════════════════════════════════
function Get-RoleDirectories {
    <#
    .SYNOPSIS
        Recursively fetches all directories for a given role.
        Returns a flat ArrayList of directory records:
          { roleId, roleName, dirId, dirName, dirPath, depth, parentDirId, parentDirName,
            permissionString, permissionCols, hasChildren, apiError }
    #>
    param(
        [string]$RoleId,
        [string]$RoleName
    )

    $allDirs = [System.Collections.ArrayList]@()

    # Stack-based iterative DFS: each item = @{ parentDirId=""; depth=0; parentPath="" }
    $stack = [System.Collections.Stack]::new()
    $stack.Push(@{ parentDirId = $null; depth = 0; parentPath = "" ; parentDirName = "" })

    $visitedDirIds = [System.Collections.Generic.HashSet[string]]@()

    while ($stack.Count -gt 0) {
        $frame = $stack.Pop()
        $parentDirId   = $frame.parentDirId
        $depth         = $frame.depth
        $parentPath    = $frame.parentPath
        $parentDirName = $frame.parentDirName

        # Build the correct URL
        if ($null -eq $parentDirId -or $parentDirId -eq "") {
            # Top-level directories for this role
            $url = "$CR_URL/v1/repository/role/$RoleId/directories"
            $desc = "GetDirs/role=$RoleId"
        } else {
            # Subdirectories of a specific directory
            $url = "$CR_URL/v1/repository/role/$RoleId/directories/$parentDirId/directories"
            $desc = "GetSubDirs/role=$RoleId/dir=$parentDirId"
        }

        $rawResponse = $null
        $apiError    = ""

        try {
            $rawResponse = Invoke-ApiWithRetry -Uri $url -Method "GET" -Description $desc
            Start-Sleep -Milliseconds $DELAY_MS
        } catch {
            $statusCode = 0
            if ($_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
            }
            $apiError = "HTTP $statusCode : $($_.Exception.Message)"
            Write-Log "  WARN: $desc failed -- $apiError" -Level WARN
            [void]$script:FailedApiCalls.Add([PSCustomObject]@{
                roleId     = $RoleId
                roleName   = $RoleName
                dirId      = $parentDirId
                apiDesc    = $desc
                error      = $apiError
                timestamp  = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            })
            continue
        }

        # DEBUG: dump raw response for first role
        if ($DEBUG_DUMP -and $depth -eq 0 -and $script:RoleCounter -eq 1) {
            $dumpPath = "debug_role_${RoleId}_dirs.json"
            $rawResponse | ConvertTo-Json -Depth 10 | Out-File $dumpPath -Encoding UTF8
            Write-Log "  [DEBUG] Raw directory response dumped -> $dumpPath"
        }

        # Normalise response -- handle both array and {list:[...]} shapes
        $dirList = @()
        if ($rawResponse -is [array]) {
            $dirList = $rawResponse
        } elseif ($rawResponse.list) {
            $dirList = @($rawResponse.list)
        } elseif ($rawResponse.directories) {
            $dirList = @($rawResponse.directories)
        } elseif ($rawResponse -is [PSCustomObject]) {
            # Single object returned -- wrap it
            $dirList = @($rawResponse)
        }

        foreach ($dir in $dirList) {
            $dirId   = [string]$dir.id
            $dirName = [string]$dir.name
            $dirPath = if ($dir.path) { [string]$dir.path } `
                       elseif ($parentPath) { "$parentPath/$dirName" } `
                       else { "/$dirName" }

            # Cycle guard
            if ($dirId -and $visitedDirIds.Contains($dirId)) {
                Write-Log "  [CycleGuard] Skipping already-visited dir: $dirName (id=$dirId)" -Level DEBUG
                continue
            }
            if ($dirId) { [void]$visitedDirIds.Add($dirId) }

            $permString = Get-PermissionString  -DirObj $dir
            $permCols   = Get-PermissionColumns -DirObj $dir

            # Check if this directory has children
            $hasChildren = $false
            if ($null -ne $dir.hasChildren)  { $hasChildren = [bool]$dir.hasChildren }
            elseif ($null -ne $dir.children) { $hasChildren = ($dir.children -gt 0) }

            $dirRecord = [PSCustomObject]@{
                roleId         = $RoleId
                roleName       = $RoleName
                dirId          = $dirId
                dirName        = $dirName
                dirPath        = $dirPath
                depth          = $depth
                parentDirId    = if ($parentDirId) { $parentDirId } else { "" }
                parentDirName  = $parentDirName
                permString     = $permString
                permCols       = $permCols
                hasChildren    = $hasChildren
                apiError       = $apiError
            }

            [void]$allDirs.Add($dirRecord)
            $script:DirCounter++

            Write-Log ("    Dir #$($script:DirCounter): $dirName  (depth=$depth, id=$dirId, perms=$permString)")

            # Push subdirectory frame onto stack for further traversal
            if ($dirId) {
                $stack.Push(@{
                    parentDirId   = $dirId
                    depth         = $depth + 1
                    parentPath    = $dirPath
                    parentDirName = $dirName
                })
            }
        }
    }

    return ,$allDirs
}


# ═══════════════════════════════════════════════════════════════════════════════
#  EXCEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
function Set-HeaderRow {
    param($Ws, [string[]]$Headers, [string]$BgColor = "2E75B6")
    for ($c = 1; $c -le $Headers.Count; $c++) {
        $cell = $Ws.Cells[1, $c]
        $cell.Value = $Headers[$c - 1]
        $cell.Style.Font.Bold = $true
        $cell.Style.Font.Color.SetColor([System.Drawing.Color]::White)
        $cell.Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
        $cell.Style.Fill.BackgroundColor.SetColor([System.Drawing.ColorTranslator]::FromHtml("#$BgColor"))
        $cell.Style.HorizontalAlignment = [OfficeOpenXml.Style.ExcelHorizontalAlignment]::Center
    }
}

function Set-CellValue {
    param($Cell, $Value)
    if ($null -eq $Value) { $Cell.Value = ""; return }
    $Cell.Value = [string]$Value
}

function Auto-FitColumns {
    param($Ws, [int]$MaxCols)
    try {
        $Ws.Cells[1, 1, $Ws.Dimension.End.Row, $MaxCols].AutoFitColumns()
    } catch {}
}


# ═══════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL EXCEL  --  three-phase approach
#    Initialize-Excel   : called once at start -- creates file + headers
#    Flush-RoleToExcel  : called after each role -- appends role row + dir rows
#    Finalize-Excel     : called at end -- writes Scan Issues, auto-fits, saves
# ═══════════════════════════════════════════════════════════════════════════════

$script:RolesHeaders     = @("Role ID","Role Name","Description","Created By","Created On",
                              "Last Modified By","Last Modified On","Directory Count","User Count")
$script:StaticDirHeaders = @("Role ID","Role Name","Directory ID","Directory Name",
                              "Path","Depth","Parent ID",
                              "Run and Schedule","Check In","Check Out",
                              "View Content","Clone","Delete from Public","Manage")
$script:IssueHeaders     = @("Role ID","Role Name","Directory ID","API Called","Error","Timestamp")
$script:TotalDirCols     = 14

function Get-ExcelPackage {
    <# Opens existing file or creates new package. Always call Dispose after use. #>
    param([string]$OutPath)
    if (-not (Get-Module -Name ImportExcel -ErrorAction SilentlyContinue)) {
        Import-Module ImportExcel -ErrorAction Stop
    }
    try { [OfficeOpenXml.ExcelPackage]::LicenseContext = [OfficeOpenXml.LicenseContext]::NonCommercial } catch {}

    if (Test-Path $OutPath) {
        return New-Object OfficeOpenXml.ExcelPackage ([System.IO.FileInfo]::new($OutPath))
    } else {
        return New-Object OfficeOpenXml.ExcelPackage
    }
}

function Save-ExcelPackage {
    param($Epkg, [string]$OutPath)
    for ($sa = 1; $sa -le 3; $sa++) {
        try {
            $Epkg.SaveAs([System.IO.FileInfo]::new($OutPath))
            return
        } catch {
            Write-Log "SaveAs attempt $sa/3 failed: $($_.Exception.Message)" -Level WARN
            if ($sa -lt 3) { Start-Sleep -Seconds 5 }
        }
    }
}

function Get-InputRoleIds {
    <#
    .SYNOPSIS
        Reads the "Role IDs" sheet (column A, from row 2) from the output file.
        Returns an array of trimmed non-empty strings, or an empty array if:
          - the file does not exist
          - the sheet does not exist
          - the sheet has no data rows
    #>
    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
    if (-not (Test-Path $outPath)) {
        Write-Log "Input file not found -- will process ALL roles." -Level INFO
        return @()
    }

    $epkg = $null
    $ids  = [System.Collections.Generic.List[string]]@()
    try {
        $epkg = Get-ExcelPackage -OutPath $outPath
        $ws   = $epkg.Workbook.Worksheets["Role IDs"]
        if ($null -eq $ws -or $null -eq $ws.Dimension) {
            Write-Log "'Role IDs' sheet is empty or missing -- will process ALL roles." -Level INFO
            return @()
        }
        $lastRow = $ws.Dimension.End.Row
        for ($r = 2; $r -le $lastRow; $r++) {
            $v = [string]$ws.Cells[$r, 1].Value
            if ($v.Trim() -ne "") { $ids.Add($v.Trim()) }
        }
    } catch {
        Write-Log "Could not read 'Role IDs' sheet: $($_.Exception.Message) -- will process ALL roles." -Level WARN
    } finally {
        if ($null -ne $epkg) { try { $epkg.Dispose() } catch {} }
    }

    if ($ids.Count -gt 0) {
        Write-Log "Input filter: $($ids.Count) role ID(s) found in 'Role IDs' sheet."
    } else {
        Write-Log "'Role IDs' sheet has no entries -- will process ALL roles." -Level INFO
    }
    return ,$ids.ToArray()
}

function Initialize-Excel {
    <#
    .SYNOPSIS
        Creates (or resets) the output workbook.
        - If the file does not exist: creates it with a 'Role IDs' input sheet + 3 output sheets.
        - If the file exists: preserves the 'Role IDs' sheet, removes and recreates the 3 output sheets.
    #>
    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
    Write-Log "Initialising Excel workbook: $outPath"

    $epkg = Get-ExcelPackage -OutPath $outPath

    # ── Ensure 'Role IDs' input sheet exists (first position) ────────────────
    $wsInput = $epkg.Workbook.Worksheets["Role IDs"]
    if ($null -eq $wsInput) {
        # Insert at position 1 so it is always the first sheet
        $wsInput = $epkg.Workbook.Worksheets.Add("Role IDs")
        $epkg.Workbook.Worksheets.MoveToStart("Role IDs")

        # Header row
        $wsInput.Cells[1, 1].Value = "Role ID"
        $wsInput.Cells[1, 2].Value = "Notes (optional)"
        $wsInput.Cells[1, 1].Style.Font.Bold = $true
        $wsInput.Cells[1, 2].Style.Font.Bold = $true
        $wsInput.Cells[1, 1].Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
        $wsInput.Cells[1, 1].Style.Fill.BackgroundColor.SetColor([System.Drawing.ColorTranslator]::FromHtml("#FFC000"))
        $wsInput.Cells[1, 2].Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
        $wsInput.Cells[1, 2].Style.Fill.BackgroundColor.SetColor([System.Drawing.ColorTranslator]::FromHtml("#FFC000"))

        # Instruction comment in row 2
        $wsInput.Cells[2, 1].Value = "# Enter one Role ID per row. Leave empty to scan ALL roles."
        $wsInput.Cells[2, 1].Style.Font.Italic = $true
        $wsInput.Cells[2, 1].Style.Font.Color.SetColor([System.Drawing.Color]::Gray)

        try { $wsInput.Column(1).Width = 20 } catch {}
        try { $wsInput.Column(2).Width = 40 } catch {}

        Write-Log "  Created 'Role IDs' input sheet."
    }

    # ── Remove existing output sheets so they are clean for this run ──────────
    foreach ($sheetName in @("Roles","Directories","Scan Issues")) {
        $existing = $epkg.Workbook.Worksheets[$sheetName]
        if ($null -ne $existing) {
            $epkg.Workbook.Worksheets.Delete($existing)
            Write-Log "  Cleared existing '$sheetName' sheet." -Level DEBUG
        }
    }

    # ── Re-create output sheets ───────────────────────────────────────────────
    $wsR = $epkg.Workbook.Worksheets.Add("Roles")
    Set-HeaderRow -Ws $wsR -Headers $script:RolesHeaders -BgColor "2E75B6"
    $wsR.View.FreezePanes(2, 1)

    $wsD = $epkg.Workbook.Worksheets.Add("Directories")
    Set-HeaderRow -Ws $wsD -Headers $script:StaticDirHeaders -BgColor "375623"
    $wsD.View.FreezePanes(2, 1)

    $wsI = $epkg.Workbook.Worksheets.Add("Scan Issues")
    Set-HeaderRow -Ws $wsI -Headers $script:IssueHeaders -BgColor "C00000"
    $wsI.View.FreezePanes(2, 1)

    Save-ExcelPackage -Epkg $epkg -OutPath $outPath
    try { $epkg.Dispose() } catch {}
    [System.GC]::Collect()

    Write-Log "Workbook initialised: 'Role IDs' sheet preserved, 3 output sheets created."
}

function Flush-RoleToExcel {
    <#
    .SYNOPSIS
        Opens the workbook, appends one role's row + all its directory rows, saves and closes.
        Directories sheet has a role-separator banner row followed by one row per directory.
        Permission columns use ✓ tick marks (green) or blank.
        Directory names are indented to show folder hierarchy.
    #>
    param(
        $Role,
        [System.Collections.ArrayList]$Dirs
    )

    $outPath  = Join-Path (Get-Location).Path $OUTPUT_FILE
    $epkg     = $null
    $permKeys = @($script:PermFieldMap.Keys)   # fixed ordered list: canRun … canManage

    try {
        $epkg = Get-ExcelPackage -OutPath $outPath
        $wsR  = $epkg.Workbook.Worksheets["Roles"]
        $wsD  = $epkg.Workbook.Worksheets["Directories"]

        # ── Write role row (Roles sheet) ──────────────────────────────────────
        $rRow = $script:Excel_RoleRow
        $wsR.Cells[$rRow, 1].Value = [string]$Role.id
        $wsR.Cells[$rRow, 2].Value = [string]$Role.name
        $wsR.Cells[$rRow, 3].Value = [string]$Role.description
        $wsR.Cells[$rRow, 4].Value = [string]$Role.createdBy
        $wsR.Cells[$rRow, 5].Value = [string]$Role.createdOn
        $wsR.Cells[$rRow, 6].Value = [string]$Role.updatedBy
        $wsR.Cells[$rRow, 7].Value = [string]$Role.updatedOn
        $wsR.Cells[$rRow, 8].Value = $Dirs.Count
        $wsR.Cells[$rRow, 9].Value = if ($null -ne $Role.countUsers) { [int]$Role.countUsers } else { "" }

        if ($rRow % 2 -eq 0) {
            for ($c = 1; $c -le $script:RolesHeaders.Count; $c++) {
                $wsR.Cells[$rRow, $c].Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
                $wsR.Cells[$rRow, $c].Style.Fill.BackgroundColor.SetColor([System.Drawing.ColorTranslator]::FromHtml("#EBF3FB"))
            }
        }
        $script:Excel_RoleRow++

        # ── Role separator banner row (Directories sheet) ─────────────────────
        $sepRow = $script:Excel_DirRow
        $wsD.Cells[$sepRow, 1, $sepRow, $script:TotalDirCols].Merge = $true
        $wsD.Cells[$sepRow, 1].Value = "  Role: $($Role.name)   (ID: $($Role.id))   [$($Dirs.Count) director(ies)]"
        $wsD.Cells[$sepRow, 1].Style.Font.Bold = $true
        $wsD.Cells[$sepRow, 1].Style.Font.Color.SetColor([System.Drawing.Color]::White)
        $wsD.Cells[$sepRow, 1].Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
        $wsD.Cells[$sepRow, 1].Style.Fill.BackgroundColor.SetColor([System.Drawing.ColorTranslator]::FromHtml("#2E75B6"))
        $wsD.Cells[$sepRow, 1].Style.VerticalAlignment = [OfficeOpenXml.Style.ExcelVerticalAlignment]::Center
        $wsD.Row($sepRow).Height = 18
        $script:Excel_DirRow++

        # ── Write directory rows ──────────────────────────────────────────────
        foreach ($d in $Dirs) {
            $dRow = $script:Excel_DirRow

            # Hierarchy indent: root = no prefix, depth 1 = "  └ ", depth 2 = "      └ " etc.
            $prefix = if ($d.depth -eq 0) { "" } `
                      else { ("    " * ($d.depth - 1)) + "  └ " }
            $displayName = $prefix + $d.dirName

            # Alternating row background (light stripe)
            if ($dRow % 2 -eq 0) {
                for ($c = 1; $c -le $script:TotalDirCols; $c++) {
                    $wsD.Cells[$dRow, $c].Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
                    $wsD.Cells[$dRow, $c].Style.Fill.BackgroundColor.SetColor([System.Drawing.ColorTranslator]::FromHtml("#F2F2F2"))
                }
            }

            $wsD.Cells[$dRow, 1].Value  = $d.roleId
            $wsD.Cells[$dRow, 2].Value  = $d.roleName
            $wsD.Cells[$dRow, 3].Value  = $d.dirId
            $wsD.Cells[$dRow, 4].Value  = $displayName
            $wsD.Cells[$dRow, 5].Value  = $d.dirPath
            $wsD.Cells[$dRow, 6].Value  = $d.depth
            $wsD.Cells[$dRow, 7].Value  = $d.parentDirId

            # Permission columns (cols 8-14): True (green) or False (red)
            for ($pi = 0; $pi -lt $permKeys.Count; $pi++) {
                $key    = $permKeys[$pi]
                $hasIt  = $d.permCols.Contains($key) -and [bool]$d.permCols[$key]
                $colIdx = 8 + $pi   # cols 8..14
                $cell   = $wsD.Cells[$dRow, $colIdx]
                $cell.Style.HorizontalAlignment = [OfficeOpenXml.Style.ExcelHorizontalAlignment]::Center

                if ($hasIt) {
                    $cell.Value = "True"
                    $cell.Style.Font.Bold  = $true
                    $cell.Style.Font.Color.SetColor([System.Drawing.Color]::FromArgb(0, 112, 0))
                } else {
                    $cell.Value = "False"
                    $cell.Style.Font.Color.SetColor([System.Drawing.Color]::FromArgb(192, 0, 0))
                }
            }

            $script:Excel_DirRow++
        }

        Save-ExcelPackage -Epkg $epkg -OutPath $outPath
        Write-Log "  [Excel] Flushed role '$($Role.name)' -- $($Dirs.Count) dir(s) written."

    } catch {
        Write-Log "ERROR in Flush-RoleToExcel for '$($Role.name)': $($_.Exception.Message)" -Level ERROR
    } finally {
        if ($null -ne $epkg) { try { $epkg.Dispose() } catch {} }
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }
}

function Finalize-Excel {
    <#
    .SYNOPSIS
        Writes the Scan Issues sheet, auto-fits all columns, saves final file.
    #>
    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
    Write-Log "Finalising Excel report: $outPath"
    $epkg = $null

    try {
        $epkg = Get-ExcelPackage -OutPath $outPath

        # ── Scan Issues ───────────────────────────────────────────────────────
        $wsI = $epkg.Workbook.Worksheets["Scan Issues"]
        if ($script:FailedApiCalls.Count -eq 0) {
            $wsI.Cells[2, 1].Value = "No issues -- all API calls succeeded."
            $wsI.Cells[2, 1].Style.Font.Italic = $true
        } else {
            $row = 2
            foreach ($f in $script:FailedApiCalls) {
                $wsI.Cells[$row, 1].Value = $f.roleId
                $wsI.Cells[$row, 2].Value = $f.roleName
                $wsI.Cells[$row, 3].Value = $f.dirId
                $wsI.Cells[$row, 4].Value = $f.apiDesc
                $wsI.Cells[$row, 5].Value = $f.error
                $wsI.Cells[$row, 6].Value = $f.timestamp
                $row++
            }
        }

        # ── Auto-fit all sheets ───────────────────────────────────────────────
        $wsR = $epkg.Workbook.Worksheets["Roles"]
        $wsD = $epkg.Workbook.Worksheets["Directories"]
        try { $wsR.Cells[$wsR.Dimension.Address].AutoFitColumns() } catch {}
        try { $wsD.Cells[$wsD.Dimension.Address].AutoFitColumns() } catch {}
        try { $wsI.Cells[$wsI.Dimension.Address].AutoFitColumns() } catch {}

        Save-ExcelPackage -Epkg $epkg -OutPath $outPath
        Write-Log "Report finalised and saved: $outPath"

    } catch {
        Write-Log "ERROR in Finalize-Excel: $($_.Exception.Message)" -Level ERROR
    } finally {
        if ($null -ne $epkg) { try { $epkg.Dispose() } catch {} }
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
function Invoke-Main {

    # ── Disable QuickEdit Mode ────────────────────────────────────────────────
    try {
        $qe = Add-Type -MemberDefinition '
            [DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint m);
            [DllImport("kernel32.dll")] public static extern bool SetConsoleMode(IntPtr h, uint m);
            [DllImport("kernel32.dll")] public static extern IntPtr GetStdHandle(int n);
        ' -Name 'QuickEditRDS' -Namespace '' -PassThru -ErrorAction Stop
        $handle = $qe::GetStdHandle(-10)
        $mode   = 0u
        $qe::GetConsoleMode($handle, [ref]$mode) | Out-Null
        $qe::SetConsoleMode($handle, ($mode -band (-bnot 0x40))) | Out-Null
        Write-Log "QuickEdit Mode disabled."
    } catch {
        Write-Log "Could not disable QuickEdit Mode (non-fatal): $($_.Exception.Message)" -Level WARN
    }

    # ── Prevent sleep ─────────────────────────────────────────────────────────
    try {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class SleepHelperRDS {
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS      = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
}
'@ -ErrorAction Stop
        [SleepHelperRDS]::SetThreadExecutionState(
            [SleepHelperRDS]::ES_CONTINUOUS -bor [SleepHelperRDS]::ES_SYSTEM_REQUIRED) | Out-Null
        Write-Log "Sleep prevention active."
    } catch {
        Write-Log "Could not set sleep prevention (non-fatal): $($_.Exception.Message)" -Level WARN
    }

    # ── Run-start banner ──────────────────────────────────────────────────────
    Write-LogBanner ""
    Write-LogBanner "════════════════════════════════════════════════════════════════"
    Write-LogBanner "  AA Role & Directory Permission Scanner  --  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-LogBanner "  Output : $OUTPUT_FILE"
    Write-LogBanner "  Log    : $LOG_FILE"
    Write-LogBanner "════════════════════════════════════════════════════════════════"

    # ── Pre-flight ────────────────────────────────────────────────────────────
    Assert-Config
    Test-OutputFileLock

    # ── Read optional role ID filter from 'Role IDs' input sheet ─────────────
    $inputRoleIds = Get-InputRoleIds   # empty array = process all

    # ── Authenticate ──────────────────────────────────────────────────────────
    Invoke-AAAuthenticate

    # ── Fetch all roles ───────────────────────────────────────────────────────
    Write-Log "Fetching all roles..."
    $allRoles = Get-AllRoles
    Write-Log "Total roles found in Control Room: $($allRoles.Count)"

    # ── Apply filter if input IDs were provided ────────────────────────────────
    if ($inputRoleIds.Count -gt 0) {
        $idSet = [System.Collections.Generic.HashSet[string]]$inputRoleIds
        $roles = @($allRoles | Where-Object { $idSet.Contains([string]$_.id) })
        Write-Log "Filtered to $($roles.Count) role(s) based on 'Role IDs' input sheet."
        # Warn about any input IDs that weren't found
        foreach ($rid in $inputRoleIds) {
            if (-not ($allRoles | Where-Object { [string]$_.id -eq $rid })) {
                Write-Log "  WARN: Role ID '$rid' from input sheet not found in Control Room." -Level WARN
            }
        }
    } else {
        $roles = $allRoles
        Write-Log "No filter applied -- processing all $($roles.Count) role(s)."
    }

    # ── Initialise Excel file with headers ───────────────────────────────────
    Initialize-Excel

    # ── Walk each role's directory tree ───────────────────────────────────────
    $totalDirs = 0

    foreach ($role in $roles) {
        $roleId   = [string]$role.id
        $roleName = [string]$role.name
        $script:RoleCounter++

        Write-Log "Role #$($script:RoleCounter)/$($roles.Count): $roleName  (id=$roleId)"

        $dirs = Get-RoleDirectories -RoleId $roleId -RoleName $roleName
        $totalDirs += $dirs.Count

        Flush-RoleToExcel -Role $role -Dirs $dirs

        Write-Log "  -> $($dirs.Count) director(ies) found for '$roleName' [Excel updated]"
    }

    Write-Log ""
    Write-Log "Scan complete -- $($roles.Count) roles, $totalDirs total directories."
    Write-Log "Failed API calls: $($script:FailedApiCalls.Count)"

    # ── Finalise report (issues sheet + auto-fit) ─────────────────────────────
    Finalize-Excel

    # ── Release sleep prevention ──────────────────────────────────────────────
    try {
        [SleepHelperRDS]::SetThreadExecutionState([SleepHelperRDS]::ES_CONTINUOUS) | Out-Null
        Write-Log "Sleep prevention released."
    } catch {}

    Write-LogBanner ""
    Write-LogBanner "Done. Open: $OUTPUT_FILE"
    Write-LogBanner "  Sheets: Roles | Directories | Scan Issues"
}


# ── Entry point ──
try {
    Invoke-Main
} catch {
    Write-Host ""
    Write-Host "FATAL ERROR: $_" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    try {
        $logPath = Join-Path (Get-Location).Path $LOG_FILE
        "FATAL ERROR: $_"      | Out-File -FilePath $logPath -Append -Encoding UTF8
        $_.ScriptStackTrace    | Out-File -FilePath $logPath -Append -Encoding UTF8
    } catch {}
    exit 1
}
