<#
.SYNOPSIS
    Automation Anywhere 360 (v30) -- Bot Inventory & Relationship Report
    Production-grade: enterprise retry/re-auth, batch incremental Excel, full edge-case handling.

.DESCRIPTION
    APIs used:
      1. POST /v2/authentication              -> bearer token
      2. POST /v2/repository/folders/{id}/list -> paginated folder contents
      3. GET  /v2/repository/files/{id}/dependencies -> child bot IDs
      4. GET  /v2/repository/files/{id}/content      -> packages + RunTask nodes

    Output:  AA_Bot_Report.xlsx  (9 sheets)

    Requirements:
        Install-Module ImportExcel -Scope CurrentUser

    Usage:
        .\BotInventory_and_relationship_scanner.ps1
        -- or override via environment variables (AA_CR_URL, AA_USERNAME, AA_PASSWORD, AA_FOLDER_ID)

.NOTES
    Enhancement summary (business logic UNCHANGED):

    [RELIABILITY]
      Assert-Config              -- validates all 4 config values before any work begins;
                                    fails fast with a clear message instead of a cryptic API error.
      Test-OutputFileLock        -- checks the output .xlsx is writable before a 20-hour run starts.
      Invoke-AAAuthenticate      -- retry loop for transient 5xx/timeout auth failures (3 attempts).
      Invoke-ApiWithRetry        -- re-auth circuit breaker: pause at 3 consecutive failures,
                                    hard abort at MAX_REAUTH_FAILURES=5.
      Flush-IncrementalBatches   -- SaveAs retried 3x with 10s delay; buffers only cleared on success
                                    (falls back to clearing after all retries to prevent memory exhaustion).

    [DATA ACCURACY]
      Add-FailedApiCall          -- central tracker: every non-404 API failure is recorded with
                                    fileId, apiType, HTTP status, error message, timestamp.
      FailedFolders tracker      -- records every folder that could not be fully listed, so
                                    missing-bot impact is visible in the report.
      dataComplete / dataIssues  -- every bot record carries a boolean flag and a list of issues;
                                    callers use FailedApiCalls-count-before/after to detect failures.
      Data Status column (col 14)-- "All Bots" sheet gains a 14th column; orange background marks
                                    any bot whose content or dependencies API call failed.
      Add-SheetScanIssues        -- 9th sheet: Issue Type | Name | Path | ID | Failed API |
                                    HTTP Status | Error Message | Impact | Timestamp.
                                    Shows "No issues" when all APIs succeeded.

    [OBSERVABILITY]
      Write-RunSummary           -- final console block: total bots, complete, partial, success
                                    rate, failed folders, failed API calls.
      Excel row-limit guard      -- warns if bot or package row counts approach 1,048,576.

    [ORIGINAL ENHANCEMENTS (business logic UNCHANGED)]
      Invoke-ApiWithRetry        -- centralised retry wrapper; auto re-authenticates on 401;
                                    exponential backoff on 5xx/429/timeout.
      Initialize-IncrementalExcel -- creates workbook up-front; headers for 3 incremental sheets.
      Flush-IncrementalBatches   -- periodic batch append; saves to disk every BATCH_SIZE bots.
      Get-OrCreateWorksheet      -- works in both incremental and legacy mode.
      Invoke-GenerateReport      -- opens existing incremental file; writes only 5 aggregate sheets.
#>

#Requires -Version 5.1

[CmdletBinding()]
param()

# Force UTF-8 output encoding
try   { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8

# ─────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────
$CR_URL      = if ($env:AA_CR_URL)    { $env:AA_CR_URL }    else { "https://aa-support-5.my.automationanywhere.digital/" }
$USERNAME    = if ($env:AA_USERNAME)  { $env:AA_USERNAME }  else { "sikha.p@automationanywhere.com" }
$PASSWORD    = if ($env:AA_PASSWORD)  { $env:AA_PASSWORD }  else { "Password@123" }
$ROOT_FOLDER = if ($env:AA_FOLDER_ID) { $env:AA_FOLDER_ID } else { "14" }

# Sanitise: strip accidental whitespace / newlines from all config strings
$CR_URL      = $CR_URL.Trim()
$USERNAME    = $USERNAME.Trim()
$PASSWORD    = $PASSWORD.Trim()
$ROOT_FOLDER = $ROOT_FOLDER.Trim()
$OUTPUT_FILE = "AS_Bot_Report.xlsx"
$PAGE_SIZE   = 100                   # items per page (max 100)
$DELAY       = 0.15                  # seconds between API calls
$VERIFY_SSL  = $true                 # set $false for self-signed certs

# Set to $true on first run to dump one bot's raw content JSON for inspection
$DEBUG_DUMP_CONTENT = $false

# ═══════════════════════════════════════════════════════════════════════════════
#  RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
$MAX_RETRIES        = 3      # retry attempts after the initial failure
$RETRY_DELAY_SEC    = 5      # base wait (seconds) before first retry
$RETRY_BACKOFF_MULT = 2.0    # exponential backoff multiplier
                              #   attempt 1 -> 5 s,  attempt 2 -> 10 s,  attempt 3 -> 20 s

# ═══════════════════════════════════════════════════════════════════════════════
#  RE-AUTH CIRCUIT BREAKER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
$MAX_REAUTH_FAILURES = 5      # abort the entire run after this many consecutive re-auth failures
$REAUTH_PAUSE_SEC    = 300    # pause (seconds) when 3 consecutive re-auths fail before continuing

# ═══════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL EXCEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
$BATCH_SIZE = 20             # flush bot+folder buffers to Excel every N bots discovered


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
        "DEBUG" { Write-Host $line -ForegroundColor Gray   }
        default { Write-Host $line -ForegroundColor Cyan   }
    }
}


# ─────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────
$script:BotCounter    = 0
$script:FolderCounter = 0
$script:AuthHeaders   = @{}

# ── Re-auth circuit breaker state ────────────────────────────────────────────
$script:ConsecutiveReAuthFailures = 0

# ── Failure tracking (populated by API functions; consumed by Add-SheetScanIssues) ──
$script:FailedApiCalls  = [System.Collections.ArrayList]@()  # {fileId,apiType,error,statusCode,timestamp}
$script:FailedFolders   = [System.Collections.ArrayList]@()  # {folderId,folderPath,error,statusCode,timestamp}

# ── Incremental Excel state ───────────────────────────────────────────────────
$script:BotBatchBuffer    = [System.Collections.ArrayList]@()
$script:FolderBatchBuffer = [System.Collections.ArrayList]@()
$script:Excel_BotRow      = 2
$script:Excel_PkgRow      = 2
$script:Excel_FolderRow   = 2
$script:IncrementalMode   = $false
$script:BatchCount        = 0

# ── Tree/FullView row counters (used inside Add-Sheet* functions) ─────────────
$script:TreeRow = 2
$script:FvRow   = 2

# ── On-demand child fetch cache ───────────────────────────────────────────────
# Prevents duplicate API calls when Resolve-BotSubtree pre-fetches a child that
# the folder walk later encounters naturally.  Keyed by bot ID.
$script:BotCache     = @{}
$script:FullView_Row = 2      # next write row for incremental Full View

# ── Checkpoint / Resume state ─────────────────────────────────────────────────
# Checkpoint file lives alongside the output Excel.  Written after every batch
# flush.  On startup the user is offered a resume prompt if the file exists.
$script:CheckpointPath    = "AA_Bot_Report_checkpoint.json"
$script:CompletedFolders  = [System.Collections.Generic.HashSet[string]]@()
$script:RestoredBotIds    = [System.Collections.Generic.HashSet[string]]@()
$script:RestoredFolderIds = [System.Collections.Generic.HashSet[string]]@()


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
        [System.Net.ServicePointManager]::SecurityProtocol  =
            [System.Net.SecurityProtocolType]::Tls12
    }
    # PS 7+: -SkipCertificateCheck is added per-call below
}


# ─────────────────────────────────────────────────────
#  HELPER: build Invoke-RestMethod param splat
# ─────────────────────────────────────────────────────
function Get-IrmParams {
    param(
        [string]$Uri,
        [string]$Method,
        [string]$Body,
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


# ─────────────────────────────────────────────────────
#  FAILURE TRACKER HELPER
# ─────────────────────────────────────────────────────
function Add-FailedApiCall {
    <#
    .SYNOPSIS
        Records a failed API call to the central tracker.
        Called by Get-BotContent, Get-BotChildren, and Invoke-ListFolderContents
        whenever a non-recoverable, non-404 error occurs.
    #>
    param(
        [string]$FileId,
        [string]$ApiType,      # "GetContent" | "GetDependencies" | "ListFolder"
        [string]$Error,
        [int]$StatusCode = 0
    )
    [void]$script:FailedApiCalls.Add([PSCustomObject]@{
        fileId     = $FileId
        apiType    = $ApiType
        error      = $Error
        statusCode = $StatusCode
        timestamp  = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    })
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

function Assert-Config {
    <#
    .SYNOPSIS
        Validates all four configuration values before any API work begins.
        Fails fast with a descriptive error rather than a cryptic API failure
        20 minutes into a long run.
    #>
    $errors = [System.Collections.ArrayList]@()

    # CR_URL
    if ([string]::IsNullOrWhiteSpace($CR_URL) -or $CR_URL -match '--') {
        [void]$errors.Add("CR_URL is not configured (still has placeholder or is empty). Set the AA_CR_URL environment variable.")
    } elseif (-not $CR_URL.StartsWith("https://")) {
        [void]$errors.Add("CR_URL must start with https:// (got: '$CR_URL'). Insecure or malformed URL.")
    }

    # USERNAME
    if ([string]::IsNullOrWhiteSpace($USERNAME) -or $USERNAME -match '--') {
        [void]$errors.Add("USERNAME is not configured. Set the AA_USERNAME environment variable.")
    }

    # PASSWORD
    if ([string]::IsNullOrWhiteSpace($PASSWORD) -or $PASSWORD -match '--') {
        [void]$errors.Add("PASSWORD is not configured. Set the AA_PASSWORD environment variable.")
    }

    # ROOT_FOLDER
    if ([string]::IsNullOrWhiteSpace($ROOT_FOLDER) -or $ROOT_FOLDER -match '--') {
        [void]$errors.Add("ROOT_FOLDER is not configured. Set the AA_FOLDER_ID environment variable.")
    } elseif ($ROOT_FOLDER -notmatch '^\d+$') {
        [void]$errors.Add("ROOT_FOLDER must be a numeric folder ID (got: '$ROOT_FOLDER'). Find it in the AA Control Room URL or folder settings.")
    }

    if ($errors.Count -gt 0) {
        Write-Log "Configuration validation FAILED ($($errors.Count) error(s)):" -Level ERROR
        foreach ($e in $errors) { Write-Log "  - $e" -Level ERROR }
        Write-Log "Tip: set env vars before running:  `$env:AA_CR_URL = 'https://your-cr.automationanywhere.digital'" -Level ERROR
        throw "Configuration validation failed. Correct the above errors and re-run."
    }

    Write-Log "Config OK: URL=$CR_URL  User=$USERNAME  FolderID=$ROOT_FOLDER"
}


function Test-OutputFileLock {
    <#
    .SYNOPSIS
        Verifies the output Excel file can be written to before starting a potentially
        long-running scan.  Catches the common mistake of leaving the file open in Excel.
    #>
    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE

    if (Test-Path $outPath) {
        # File exists -- try to open it exclusively to detect any lock
        try {
            $stream = [System.IO.File]::Open(
                $outPath,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None)
            $stream.Close()
            $stream.Dispose()
            Write-Log "Output file exists and is writable: $outPath"
        } catch {
            throw ("Output file is locked and cannot be written: '$outPath'. " +
                   "If it is open in Excel, close it first, then re-run. Error: $($_.Exception.Message)")
        }
    } else {
        # File does not exist yet -- verify the directory is writable
        try {
            $stream = [System.IO.File]::Open(
                $outPath,
                [System.IO.FileMode]::Create,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None)
            $stream.Close()
            $stream.Dispose()
            Remove-Item $outPath -Force -ErrorAction SilentlyContinue
            Write-Log "Output path is writable: $outPath"
        } catch {
            throw ("Cannot create output file at '$outPath'. " +
                   "Check disk space and folder permissions. Error: $($_.Exception.Message)")
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CENTRALISED RETRY WRAPPER  (with re-auth circuit breaker)
#
#  All three API call functions (Invoke-ListFolderContents, Get-BotChildren,
#  Get-BotContent) delegate here instead of calling Invoke-RestMethod directly.
#  Business logic in those callers is UNCHANGED.
#
#  Retry policy:
#    401  -> re-authenticate, then retry immediately (no sleep).
#           Circuit breaker: pause at 3 consecutive re-auth failures;
#           abort hard at MAX_REAUTH_FAILURES consecutive failures.
#    429 / 5xx / timeout (0) -> exponential backoff sleep, then retry.
#    400 / 403 / 404          -> non-retryable; re-throw so caller handles it.
#    After MAX_RETRIES         -> log ERROR and re-throw final exception.
# ═══════════════════════════════════════════════════════════════════════════════
function Invoke-ApiWithRetry {
    param(
        [string]$Uri,
        [string]$Method,
        [string]$Body        = $null,
        [string]$Description = "API call"
    )

    $totalAttempts = $MAX_RETRIES + 1   # 1 original + MAX_RETRIES retries

    for ($attempt = 1; $attempt -le $totalAttempts; $attempt++) {

        # Rebuild IRM params on every attempt so a refreshed token is always current
        $irmParams = Get-IrmParams -Uri $Uri -Method $Method `
                                   -Body $Body -Headers $script:AuthHeaders
        try {
            return Invoke-RestMethod @irmParams -ErrorAction Stop

        } catch {
            # Extract HTTP status code (works in PS 5.1 and PS 7+)
            $statusCode = 0
            if ($_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
            }
            $isLastAttempt = ($attempt -ge $totalAttempts)

            # ── 401 Unauthorized: token expired -> re-authenticate ────────────
            if ($statusCode -eq 401) {
                if ($isLastAttempt) {
                    Write-Log "[Retry] $Description -- 401 FINAL FAILURE after $MAX_RETRIES retries." -Level ERROR
                    throw
                }
                Write-Log "[Retry] $Description -- 401 token expired (attempt $attempt/$totalAttempts). Re-authenticating..." -Level WARN
                try {
                    Invoke-AAAuthenticate
                    $script:ConsecutiveReAuthFailures = 0   # reset circuit breaker on success
                    Write-Log "[Retry] Token refreshed. Retrying $Description..." -Level INFO
                } catch {
                    $script:ConsecutiveReAuthFailures++
                    $reAuthErr = $_.Exception.Message
                    Write-Log ("[CircuitBreaker] Re-auth failed ($($script:ConsecutiveReAuthFailures)/$MAX_REAUTH_FAILURES): $reAuthErr") -Level ERROR

                    # Hard abort: Control Room unreachable for too long
                    if ($script:ConsecutiveReAuthFailures -ge $MAX_REAUTH_FAILURES) {
                        throw ("CIRCUIT BREAKER OPEN: $MAX_REAUTH_FAILURES consecutive re-authentication failures. " +
                               "The Control Room appears to be down or credentials have been revoked. " +
                               "Aborting to prevent producing a report with all-empty data.")
                    }

                    # Soft pause: give the Control Room time to recover
                    if ($script:ConsecutiveReAuthFailures -ge 3) {
                        Write-Log ("[CircuitBreaker] Pausing ${REAUTH_PAUSE_SEC}s before next attempt -- " +
                                   "Control Room may be temporarily unavailable...") -Level WARN
                        Start-Sleep -Seconds $REAUTH_PAUSE_SEC
                    }
                }
                # Continue to next attempt -- no sleep after re-auth (token is immediately valid)

            # ── 429 Throttled / 5xx Server Error / Network timeout (code=0) ──
            } elseif ($statusCode -eq 429 -or
                      ($statusCode -ge 500 -and $statusCode -lt 600) -or
                      $statusCode -eq 0) {
                if ($isLastAttempt) {
                    Write-Log "[Retry] $Description -- HTTP $statusCode FINAL FAILURE after $MAX_RETRIES retries." -Level ERROR
                    throw
                }
                $delaySec = [int]([Math]::Round($RETRY_DELAY_SEC * [Math]::Pow($RETRY_BACKOFF_MULT, $attempt - 1)))
                Write-Log "[Retry] $Description -- HTTP $statusCode transient error. Backoff ${delaySec}s (attempt $attempt/$totalAttempts)..." -Level WARN
                Start-Sleep -Seconds $delaySec

            # ── 400 / 403 / 404 and anything else: non-retryable ─────────────
            } else {
                throw
            }
        }
    }
}


# ─────────────────────────────────────────────────────
#  API CALLS
# ─────────────────────────────────────────────────────

function Invoke-AAAuthenticate {
    <#
    .SYNOPSIS
        POST /v2/authentication -- obtains a bearer token.
        Does NOT go through Invoke-ApiWithRetry (to avoid infinite recursion).
        Retries up to 3 times on 5xx / network timeout; fails immediately on 401.
    #>
    $authMaxRetries = 3

    for ($authAttempt = 1; $authAttempt -le $authMaxRetries; $authAttempt++) {
        Write-Log "Authenticating to $CR_URL ... (attempt $authAttempt/$authMaxRetries)"
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
            Write-Log "Authenticated successfully."
            return

        } catch {
            $statusCode = 0
            if ($_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
            }
            $isLast = ($authAttempt -ge $authMaxRetries)

            # 401 = wrong credentials; no point retrying
            if ($statusCode -eq 401) {
                throw "Authentication failed (401 Unauthorized): Invalid credentials for user '$USERNAME'. Check AA_USERNAME and AA_PASSWORD."
            }

            # 5xx / network timeout -- retry with backoff
            if ($isLast) {
                throw "Authentication failed after $authMaxRetries attempts (HTTP $statusCode): $($_.Exception.Message)"
            }
            $delaySec = $authAttempt * 5   # 5 s, 10 s
            Write-Log "Auth attempt $authAttempt failed (HTTP $statusCode). Retrying in ${delaySec}s..." -Level WARN
            Start-Sleep -Seconds $delaySec
        }
    }
}


function Invoke-ListFolderContents {
    <#
    .SYNOPSIS
        POST /v2/repository/folders/{folderId}/list
        Paginates automatically. Returns all items (files + sub-folders).
        On failure: logs to $script:FailedFolders and returns whatever was collected so far.
    #>
    param(
        [string]$FolderId,
        [string]$FolderPath = "Unknown"    # passed for FailedFolders tracking
    )

    $url    = "$CR_URL/v2/repository/folders/$FolderId/list"
    $items  = [System.Collections.ArrayList]@()
    $offset = 0

    do {
        $body = @{
            sort = @( @{ field = "name"; direction = "asc" } )
            page = @{ offset = $offset; length = $PAGE_SIZE }
        } | ConvertTo-Json -Depth 5

        try {
            $data = Invoke-ApiWithRetry -Uri $url -Method "POST" -Body $body `
                                        -Description "ListFolder/$FolderId offset=$offset"
        } catch {
            $statusCode = 0
            if ($_.Exception.Response) {
                try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
            }
            $errMsg = $_.Exception.Message
            Write-Log "ERROR listing folder $FolderId ('$FolderPath') at offset $offset : $errMsg" -Level ERROR

            # Record the failed folder scan
            [void]$script:FailedFolders.Add([PSCustomObject]@{
                folderId   = $FolderId
                folderPath = $FolderPath
                error      = $errMsg
                statusCode = $statusCode
                timestamp  = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
            })
            break
        }

        $batch  = if ($data.list) { @($data.list) } else { @() }
        foreach ($item in $batch) { [void]$items.Add($item) }

        $total  = if ($data.page -and ($null -ne $data.page.total)) { [int]$data.page.total } else { 0 }
        $offset += $batch.Count

        if ($batch.Count -gt 0 -and $offset -lt $total) {
            Start-Sleep -Milliseconds ([int]($DELAY * 1000))
        }
    } while ($offset -lt $total -and $batch.Count -gt 0)

    return ,$items
}


function Get-BotChildren {
    <#
    .SYNOPSIS
        GET /v2/repository/files/{fileId}/dependencies
        Returns only entries where dependencyType == "SCANNED" and type == taskbot.
        On non-404 failure: logs to $script:FailedApiCalls; returns empty array.
    #>
    param([string]$FileId)

    try {
        $resp = Invoke-ApiWithRetry -Uri "$CR_URL/v2/repository/files/$FileId/dependencies" `
                                    -Method "GET" `
                                    -Description "GetDependencies/$FileId"
    } catch {
        $statusCode = 0
        if ($_.Exception.Response) {
            try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
        }
        # 400/404: bad request or not found -- silent, not a data-quality issue
        if ($statusCode -eq 400 -or $statusCode -eq 404) { return ,@() }

        # 403: insufficient permissions -- worth tracking
        # 5xx / timeout (after retries exhausted): worth tracking
        $errMsg = $_.Exception.Message
        Write-Log "WARN: dependencies API error for $FileId (HTTP $statusCode): $errMsg" -Level WARN
        Add-FailedApiCall -FileId $FileId -ApiType "GetDependencies" -Error $errMsg -StatusCode $statusCode
        return ,@()
    }

    $deps = if ($resp -is [array]) { $resp } `
            elseif ($resp.dependencies) { @($resp.dependencies) } `
            else { @() }

    $result = [System.Collections.ArrayList]@()
    foreach ($d in $deps) {
        if ($d.dependencyType -eq "SCANNED" -and
            $d.type -eq "application/vnd.aa.taskbot" -and
            [string]$d.id -ne [string]$FileId) {
            [void]$result.Add([PSCustomObject]@{
                id   = [string]$d.id
                name = [string]$d.name
                path = [string]$d.path
                type = [string]$d.type
            })
        }
    }
    return ,$result
}


function Get-BotContent {
    <#
    .SYNOPSIS
        GET /v2/repository/files/{fileId}/content
        Returns the full bot definition JSON (packages + action nodes).
        On non-404 failure: logs to $script:FailedApiCalls; returns $null.
    #>
    param([string]$FileId)

    try {
        $data = Invoke-ApiWithRetry -Uri "$CR_URL/v2/repository/files/$FileId/content" `
                                    -Method "GET" `
                                    -Description "GetContent/$FileId"
    } catch {
        $statusCode = 0
        if ($_.Exception.Response) {
            try { $statusCode = [int]$_.Exception.Response.StatusCode } catch {}
        }
        # 400/404: bad request or not found -- silent
        if ($statusCode -eq 400 -or $statusCode -eq 404) { return $null }

        # 403: no permission to read this bot's content
        # 5xx / timeout (after retries exhausted)
        $errMsg = $_.Exception.Message
        Write-Log "WARN: content API error for $FileId (HTTP $statusCode): $errMsg" -Level WARN
        Add-FailedApiCall -FileId $FileId -ApiType "GetContent" -Error $errMsg -StatusCode $statusCode
        return $null
    }

    if ($DEBUG_DUMP_CONTENT) {
        $dumpPath = "content_debug_$FileId.json"
        $data | ConvertTo-Json -Depth 20 | Out-File -FilePath $dumpPath -Encoding UTF8
        Write-Log "Raw content dumped -> $dumpPath"
    }

    return $data
}


function Get-ParsedPackages {
    <#
    .SYNOPSIS
        Extract [{name, version}] from the bot content JSON.
    #>
    param($Content)

    $pkgs = @{}

    # Primary: top-level "packages" array (A360 v30 confirmed)
    if ($Content -and $Content.packages) {
        foreach ($p in @($Content.packages)) {
            $name = [string]$p.name
            $ver  = [string]$p.version
            if ($name) { $pkgs[$name] = $ver }
        }
    }

    # Fallback for alternate key names
    if ($pkgs.Count -eq 0 -and $Content) {
        foreach ($key in @("packageInfos", "packageDependencies")) {
            $arr = $Content.$key
            if ($arr) {
                foreach ($p in @($arr)) {
                    $name = if ($p.name)            { [string]$p.name }            `
                            elseif ($p.packageName) { [string]$p.packageName }     `
                            else { "" }
                    $ver  = if ($p.version)            { [string]$p.version }         `
                            elseif ($p.packageVersion) { [string]$p.packageVersion }  `
                            else { "" }
                    if ($name) { $pkgs[$name] = $ver }
                }
            }
        }
    }

    $result = [System.Collections.ArrayList]@()
    foreach ($n in ($pkgs.Keys | Sort-Object)) {
        [void]$result.Add([PSCustomObject]@{ name = $n; version = $pkgs[$n] })
    }
    return ,$result
}


function Get-ChildrenFromContent {
    <#
    .SYNOPSIS
        Extract child bot references from the bot content JSON node tree.
        Iterative stack-based DFS. Looks for commandName="runTask" nodes.
    #>
    param($Content)

    $children  = [System.Collections.ArrayList]@()
    $seenPaths = [System.Collections.Generic.HashSet[string]]@()

    $stack = [System.Collections.Stack]::new()
    $stack.Push($Content)

    while ($stack.Count -gt 0) {
        $obj = $stack.Pop()
        if ($null -eq $obj) { continue }

        if ($obj -is [PSCustomObject]) {
            if ($obj.commandName -eq "runTask" -and $obj.packageName -eq "TaskBot") {
                foreach ($attr in @($obj.attributes)) {
                    if ($attr -and $attr.name -eq "taskbot") {
                        $val     = $attr.value
                        $tbFile  = if ($val) { $val.taskbotFile } else { $null }
                        $rawPath = if ($tbFile) { [string]$tbFile.string } else { "" }
                        if ($rawPath -and -not $seenPaths.Contains($rawPath)) {
                            [void]$seenPaths.Add($rawPath)
                            $clean = [System.Uri]::UnescapeDataString($rawPath)
                            $clean = $clean -replace '^repository:///', ''
                            $clean = $clean -replace '/', '\'
                            $botName = ($clean -split '\\')[-1]
                            [void]$children.Add([PSCustomObject]@{
                                id   = ""
                                name = $botName
                                path = $clean
                            })
                        }
                    }
                }
            }
            foreach ($prop in $obj.PSObject.Properties) {
                $stack.Push($prop.Value)
            }

        } elseif ($obj -is [System.Collections.IEnumerable] -and $obj -isnot [string]) {
            foreach ($item in $obj) {
                $stack.Push($item)
            }
        }
    }

    return ,$children
}


function Search-PackagesRecursive {
    <#
    .SYNOPSIS
        Fallback: find package name+version pairs anywhere in JSON (iterative stack-based).
    #>
    param($Obj, [hashtable]$Result)

    $stack = [System.Collections.Stack]::new()
    $stack.Push($Obj)

    while ($stack.Count -gt 0) {
        $cur = $stack.Pop()
        if ($null -eq $cur) { continue }

        if ($cur -is [PSCustomObject]) {
            $name = if ($cur.name)            { [string]$cur.name }            `
                    elseif ($cur.packageName) { [string]$cur.packageName }     `
                    else { "" }
            $ver  = if ($cur.version)            { [string]$cur.version }         `
                    elseif ($cur.packageVersion) { [string]$cur.packageVersion }  `
                    else { "" }
            if ($name -and $ver -and $ver -match '\.') {
                $Result[$name] = $ver
            }
            foreach ($prop in $cur.PSObject.Properties) {
                $stack.Push($prop.Value)
            }
        } elseif ($cur -is [System.Collections.IEnumerable] -and $cur -isnot [string]) {
            foreach ($item in $cur) { $stack.Push($item) }
        }
    }
}


# ─────────────────────────────────────────────────────
#  ON-DEMAND CHILD RESOLUTION + INCREMENTAL FULL VIEW
# ─────────────────────────────────────────────────────

$BOT_MIME = "application/vnd.aa.taskbot"


function Resolve-BotSubtree {
    <#
    .SYNOPSIS
        Recursively fetches all children (and their children) of a bot on-demand
        via API, so the complete subtree is known immediately after scanning any bot.

        Results are stored in $script:BotCache (id -> botRecord) to prevent
        duplicate API calls when the normal folder walk later encounters the same bots.

        On-demand fetched bots are added to $AllBots (so they appear in aggregate
        sheets at the end) but NOT to $script:BotBatchBuffer -- their metadata
        (depth, created_by, file_size etc.) comes from the folder listing, which
        happens later.  The folder walk's cache-hit path adds them to the buffer
        once full metadata is available.

        Depth is capped at 15 levels to prevent infinite loops on circular refs.
    #>
    param(
        $BotRecord,
        [System.Collections.ArrayList]$AllBots,
        [System.Collections.Generic.HashSet[string]]$VisitedIds,
        [int]$CurrentDepth = 0
    )

    if ($CurrentDepth -ge 15) { return }

    foreach ($childRef in @($BotRecord.children)) {
        $childId   = [string]$childRef.id
        $childName = [string]$childRef.name

        if (-not $childId) { continue }
        if ($VisitedIds.Contains($childId)) { continue }
        [void]$VisitedIds.Add($childId)

        # Already cached by a previous scan -- just recurse into its children
        if ($script:BotCache.ContainsKey($childId)) {
            Resolve-BotSubtree -BotRecord $script:BotCache[$childId] -AllBots $AllBots `
                               -VisitedIds $VisitedIds -CurrentDepth ($CurrentDepth + 1)
            continue
        }

        Write-Log "      [OnDemand] Fetching child: $childName (id=$childId)" -Level INFO

        # ── Fetch content ──────────────────────────────────────────────────────
        $fBefore  = $script:FailedApiCalls.Count
        $cContent = Get-BotContent -FileId $childId
        $cContentFailed = ($script:FailedApiCalls.Count -gt $fBefore)
        Start-Sleep -Milliseconds ([int]($DELAY * 1000))

        $cPackages            = Get-ParsedPackages      -Content $cContent
        $cChildrenFromContent = Get-ChildrenFromContent -Content $cContent

        # ── Fetch dependencies ─────────────────────────────────────────────────
        $fBefore2 = $script:FailedApiCalls.Count
        $cChildrenFromDeps = Get-BotChildren -FileId $childId
        $cDepsFailed = ($script:FailedApiCalls.Count -gt $fBefore2)
        Start-Sleep -Milliseconds ([int]($DELAY * 1000))

        # ── Merge children ─────────────────────────────────────────────────────
        $cDepByName     = @{}
        $cContentByName = @{}
        foreach ($d in $cChildrenFromDeps)    { $cDepByName[$d.name]     = $d }
        foreach ($c in $cChildrenFromContent) { $cContentByName[$c.name] = $c }

        $cAllChildNames = [System.Collections.Generic.HashSet[string]]@()
        foreach ($k in $cDepByName.Keys)     { [void]$cAllChildNames.Add($k) }
        foreach ($k in $cContentByName.Keys) { [void]$cAllChildNames.Add($k) }

        $cChildren = [System.Collections.ArrayList]@()
        foreach ($cn in $cAllChildNames) {
            $cd = if ($cDepByName.ContainsKey($cn))     { $cDepByName[$cn] }     else { $null }
            $cc = if ($cContentByName.ContainsKey($cn)) { $cContentByName[$cn] } else { $null }
            $ccId   = if ($cd -and $cd.id)   { $cd.id }   elseif ($cc -and $cc.id)   { $cc.id }   else { "" }
            $ccPath = if ($cd -and $cd.path) { $cd.path } elseif ($cc -and $cc.path) { $cc.path } else { "" }
            [void]$cChildren.Add([PSCustomObject]@{ id = $ccId; name = $cn; path = $ccPath })
        }

        # ── Data completeness ──────────────────────────────────────────────────
        $cDataIssues = [System.Collections.ArrayList]@()
        if ($cContentFailed) { [void]$cDataIssues.Add("content API failed -- package list may be incomplete") }
        if ($cDepsFailed)    { [void]$cDataIssues.Add("dependencies API failed -- child bot list may be incomplete") }

        $childRecord = [PSCustomObject]@{
            id           = $childId
            name         = if ($childName) { $childName } else { $childId }
            path         = [string]$childRef.path
            parent_id    = ""       # filled in when folder walk reaches this bot
            depth        = 0        # filled in when folder walk reaches this bot
            children     = $cChildren
            packages     = $cPackages
            created_by   = ""       # filled in when folder walk reaches this bot
            created_on   = ""
            modified_by  = ""
            modified_on  = ""
            locked       = $false
            file_size    = 0L
            dataComplete = ($cDataIssues.Count -eq 0)
            dataIssues   = $cDataIssues
        }

        # Register in cache + master list (NOT in BotBatchBuffer -- metadata incomplete)
        $script:BotCache[$childId] = $childRecord
        [void]$AllBots.Add($childRecord)

        Write-Log ("      [OnDemand] Registered: $childName  " +
                   "($($cChildren.Count) children, $($cPackages.Count) packages)") -Level INFO

        # Recurse into this child's own children
        Resolve-BotSubtree -BotRecord $childRecord -AllBots $AllBots `
                           -VisitedIds $VisitedIds -CurrentDepth ($CurrentDepth + 1)
    }
}


function Flush-FullViewSubtree {
    <#
    .SYNOPSIS
        Appends one bot's complete subtree (bot + packages + all descendants indented)
        to the Full View sheet on disk immediately after the bot is scanned.

        Uses the same Write-BotBlock recursive helper as Add-SheetFullView.
        Syncs $script:FullView_Row with $script:FvRow before/after the call so row
        positions are consistent across all incremental writes.

        The Full View sheet is cleared and rewritten correctly at the end of the run
        by Add-SheetFullView (called from Invoke-GenerateReport), which applies the
        definitive MASTER / Child role labels once all bots are known.
        The incremental version exists purely for mid-run visibility.

        SaveAs is retried up to 3 times (5 s delay) matching the pattern used by
        Flush-IncrementalBatches.
    #>
    param(
        $Bot,
        [System.Collections.ArrayList]$AllBots
    )

    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
    if (-not (Test-Path $outPath)) { return }

    $epkg = $null
    try {
        if (-not (Get-Module -Name ImportExcel)) { Import-Module ImportExcel -ErrorAction Stop }
        try { [OfficeOpenXml.ExcelPackage]::LicenseContext = [OfficeOpenXml.LicenseContext]::NonCommercial } catch {}

        $epkg = New-Object OfficeOpenXml.ExcelPackage ([System.IO.FileInfo]::new($outPath))
        $ws   = $epkg.Workbook.Worksheets["Full View"]

        # Build lookup maps from all bots known so far
        $byId_fvInc      = @{}
        $byName_fvInc    = @{}
        $byPathEnd_fvInc = @{}
        foreach ($b in $AllBots) {
            if ($b.id)   { $byId_fvInc[$b.id]     = $b }
            if ($b.name) { $byName_fvInc[$b.name]  = $b }
            $bEnd = ($b.path -split '[/\\]')[-1]
            if ($bEnd) { $byPathEnd_fvInc[$bEnd.ToLower()] = $b }
        }

        # Sync the script-level row counter used by Write-BotBlock
        $script:FvRow = $script:FullView_Row

        Write-BotBlock -Bot $Bot -Level 0 -RelLabel "MASTER" `
            -VisitedIds ([System.Collections.Generic.HashSet[string]]@($Bot.id)) `
            -Ws $ws -ById $byId_fvInc -ByName $byName_fvInc -ByPathEnd $byPathEnd_fvInc

        # Add a thin separator row between subtrees
        for ($c = 1; $c -le 7; $c++) {
            Set-CellFill   -Cell $ws.Cells[$script:FvRow, $c] -HexColor "E8EDF2"
            Set-CellBorder -Cell $ws.Cells[$script:FvRow, $c]
        }
        $ws.Row($script:FvRow).Height = 8
        $script:FvRow++

        # Sync back so the next call starts at the correct row
        $script:FullView_Row = $script:FvRow

        # SaveAs with retry
        for ($sa = 1; $sa -le 3; $sa++) {
            try {
                $epkg.SaveAs([System.IO.FileInfo]::new($outPath))
                break
            } catch {
                Write-Log "Flush-FullViewSubtree: SaveAs attempt $sa/3 failed: $($_.Exception.Message)" -Level WARN
                if ($sa -lt 3) { Start-Sleep -Seconds 5 }
            }
        }

    } catch {
        Write-Log "ERROR in Flush-FullViewSubtree for '$($Bot.name)': $($_.Exception.Message)" -Level ERROR
    } finally {
        if ($null -ne $epkg) { try { $epkg.Dispose() } catch {}; $epkg = $null }
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }
}


function Invoke-WalkFolder {
    <#
    .SYNOPSIS
        Recursively walks the folder tree.
        Populates $Bots and $Folders.  Also fills batch buffers for incremental Excel.

        Each bot record now includes:
          dataComplete (bool)   -- $false if any API call for this bot failed
          dataIssues   (list)   -- human-readable list of what failed (empty if all OK)

        The dataComplete flag is set by comparing $script:FailedApiCalls.Count
        before and after each API call -- no changes to function signatures required.
    #>
    param(
        [string]$FolderId,
        [string]$FolderPath,
        [int]$Depth,
        [System.Collections.ArrayList]$Bots,
        [System.Collections.ArrayList]$Folders,
        [System.Collections.Generic.HashSet[string]]$SeenFolders
    )

    if ($SeenFolders.Contains($FolderId)) { return }
    [void]$SeenFolders.Add($FolderId)

    # ── Checkpoint: folder fully processed in a prior run -- skip entirely ────
    if ($script:CompletedFolders.Contains($FolderId)) {
        Write-Log "   [RESUME] Skipping completed folder: $FolderPath" -Level DEBUG
        return
    }

    $script:FolderCounter++
    Write-Log "[$script:FolderCounter folders] Scanning: $FolderPath"

    # Pass FolderPath so FailedFolders tracking has human-readable context
    $items = Invoke-ListFolderContents -FolderId $FolderId -FolderPath $FolderPath
    Start-Sleep -Milliseconds ([int]($DELAY * 1000))

    Write-Log "   +-- $($items.Count) item(s) found in this folder"

    foreach ($item in $items) {
        $itemId   = [string]$item.id
        $itemName = [string]$item.name
        $itemPath = "$FolderPath/$itemName"
        $isFolder = [bool]$item.folder
        $mime     = if ($item.type)     { [string]$item.type }     `
                    elseif ($item.mimeType) { [string]$item.mimeType } `
                    else { "" }

        if ($isFolder) {
            $folderRecord = [PSCustomObject]@{
                id        = $itemId
                name      = $itemName
                path      = $itemPath
                parent_id = $FolderId
                depth     = $Depth
            }
            # Only add to lists / buffer if not already restored from checkpoint
            if (-not $script:RestoredFolderIds.Contains($itemId)) {
                [void]$Folders.Add($folderRecord)
                [void]$script:FolderBatchBuffer.Add($folderRecord)
            }

            Invoke-WalkFolder -FolderId $itemId -FolderPath $itemPath -Depth ($Depth + 1) `
                              -Bots $Bots -Folders $Folders -SeenFolders $SeenFolders

        } elseif ($mime -eq $BOT_MIME -or $mime.EndsWith("taskbot")) {

            # ── Cache hit: bot already fetched on-demand as a child of a previously
            #    scanned bot.  Update metadata from the folder listing (depth, created_by
            #    etc.) then add to the batch buffer so All Bots sheet gets full data.
            if ($script:BotCache.ContainsKey($itemId)) {
                $cachedBot             = $script:BotCache[$itemId]
                $cachedBot.parent_id   = $FolderId
                $cachedBot.depth       = $Depth
                $cachedBot.created_by  = [string]$item.createdBy
                $cachedBot.created_on  = [string]$item.createdOn
                $cachedBot.modified_by = [string]$item.lastModifiedBy
                $cachedBot.modified_on = [string]$item.lastModifiedOn
                $cachedBot.locked      = [bool]$item.locked
                $cachedBot.file_size   = if ($null -ne $item.fileSize) { [long]$item.fileSize } else { 0L }
                $script:BotCounter++

                # Bots restored from a checkpoint are already in the Excel file --
                # skip the buffer to prevent duplicate rows in incremental sheets.
                if ($script:RestoredBotIds.Contains($itemId)) {
                    Write-Log "   Bot #$($script:BotCounter): $itemName  (depth=$Depth) [checkpoint restore -- skip buffer]" -Level DEBUG
                    continue
                }

                Write-Log "   Bot #$($script:BotCounter): $itemName  (depth=$Depth) [cache hit]"
                # Add to batch buffer now that metadata is complete
                [void]$script:BotBatchBuffer.Add($cachedBot)
                if ($script:BotBatchBuffer.Count -ge $BATCH_SIZE) {
                    Flush-IncrementalBatches
                    Save-Checkpoint -Bots $Bots -Folders $Folders
                }
                continue   # skip API calls -- subtree already resolved
            }

            $script:BotCounter++
            Write-Log "   Bot #$($script:BotCounter): $itemName  (depth=$Depth)"

            # ── API 4: /content ────────────────────────────────────────────────
            $failsBefore = $script:FailedApiCalls.Count
            $content     = Get-BotContent -FileId $itemId
            $contentFailed = ($script:FailedApiCalls.Count -gt $failsBefore)
            Start-Sleep -Milliseconds ([int]($DELAY * 1000))

            $packages            = Get-ParsedPackages      -Content $content
            $childrenFromContent = Get-ChildrenFromContent -Content $content

            # ── API 3: /dependencies ───────────────────────────────────────────
            $failsBefore2     = $script:FailedApiCalls.Count
            $childrenFromDeps = Get-BotChildren -FileId $itemId
            $depsFailed       = ($script:FailedApiCalls.Count -gt $failsBefore2)
            Start-Sleep -Milliseconds ([int]($DELAY * 1000))

            # ── Determine data completeness ────────────────────────────────────
            $dataIssues = [System.Collections.ArrayList]@()
            if ($contentFailed) { [void]$dataIssues.Add("content API failed -- package list may be incomplete") }
            if ($depsFailed)    { [void]$dataIssues.Add("dependencies API failed -- child bot list may be incomplete") }
            $dataComplete = ($dataIssues.Count -eq 0)

            # ── Merge children from both sources ───────────────────────────────
            $depByName     = @{}
            $contentByName = @{}
            foreach ($d in $childrenFromDeps)    { $depByName[$d.name]     = $d }
            foreach ($c in $childrenFromContent) { $contentByName[$c.name] = $c }

            $allChildNames = [System.Collections.Generic.HashSet[string]]@()
            foreach ($k in $depByName.Keys)     { [void]$allChildNames.Add($k) }
            foreach ($k in $contentByName.Keys) { [void]$allChildNames.Add($k) }

            $children = [System.Collections.ArrayList]@()
            foreach ($cname in $allChildNames) {
                $dep = if ($depByName.ContainsKey($cname))     { $depByName[$cname] }     else { $null }
                $con = if ($contentByName.ContainsKey($cname)) { $contentByName[$cname] } else { $null }
                $cid   = if ($dep -and $dep.id)   { $dep.id }   elseif ($con -and $con.id)   { $con.id }   else { "" }
                $cpath = if ($dep -and $dep.path) { $dep.path } elseif ($con -and $con.path) { $con.path } else { "" }
                [void]$children.Add([PSCustomObject]@{
                    id   = $cid
                    name = $cname
                    path = $cpath
                })
            }

            $statusIndicator = if ($dataComplete) { "" } else { "[PARTIAL] " }
            Write-Log ("   $statusIndicator-> $($children.Count) child bot(s)  |  " +
                       "$($packages.Count) package(s)" +
                       $(if (-not $dataComplete) { "  | ISSUES: " + ($dataIssues -join "; ") } else { "" }))

            $botRecord = [PSCustomObject]@{
                id           = $itemId
                name         = $itemName
                path         = $itemPath
                parent_id    = $FolderId
                depth        = $Depth
                children     = $children
                packages     = $packages
                created_by   = [string]$item.createdBy
                created_on   = [string]$item.createdOn
                modified_by  = [string]$item.lastModifiedBy
                modified_on  = [string]$item.lastModifiedOn
                locked       = [bool]$item.locked
                file_size    = if ($null -ne $item.fileSize) { [long]$item.fileSize } else { 0L }
                dataComplete = $dataComplete
                dataIssues   = $dataIssues
            }

            # Register in cache so future encounters are cache hits
            $script:BotCache[$itemId] = $botRecord
            [void]$Bots.Add($botRecord)
            [void]$script:BotBatchBuffer.Add($botRecord)

            # ── Resolve full subtree on-demand (fetch all children/grandchildren) ──
            $subtreeVisited = [System.Collections.Generic.HashSet[string]]@($itemId)
            Resolve-BotSubtree -BotRecord $botRecord -AllBots $Bots -VisitedIds $subtreeVisited

            # ── Write this bot's complete subtree to Full View immediately ─────
            if ($script:IncrementalMode) {
                Flush-FullViewSubtree -Bot $botRecord -AllBots $Bots
            }

            if ($script:BotBatchBuffer.Count -ge $BATCH_SIZE) {
                Flush-IncrementalBatches
                Save-Checkpoint -Bots $Bots -Folders $Folders
            }

        } else {
            Write-Log "   Skipping non-bot file: $itemName ($mime)" -Level DEBUG
        }
    }

    # ── Checkpoint: mark this folder as fully processed ───────────────────────
    # Stored in memory; written to disk on the next batch flush.
    [void]$script:CompletedFolders.Add($FolderId)
}


# ─────────────────────────────────────────────────────
#  RELATIONSHIP BUILDER
# ─────────────────────────────────────────────────────

function Resolve-ChildBotFromMaps {
    param(
        [string]$ChildId,
        [string]$ChildName,
        [string]$ChildPath,
        [hashtable]$ById,
        [hashtable]$ByName,
        [hashtable]$ByPathEnd
    )
    if ($ChildId   -and $ById.ContainsKey($ChildId))         { return $ById[$ChildId] }
    if ($ChildName -and $ByName.ContainsKey($ChildName))     { return $ByName[$ChildName] }
    $end = (($ChildPath -replace '\\\\', '\') -split '[/\\]')[-1].ToLower()
    if ($ByPathEnd.ContainsKey($end))                        { return $ByPathEnd[$end] }
    return $null
}


function Recurse-Relationships {
    param(
        $ParentBot,
        [int]$Level,
        [System.Collections.Generic.HashSet[string]]$VisitedIds,
        [System.Collections.ArrayList]$Rows,
        [hashtable]$ById,
        [hashtable]$ByName,
        [hashtable]$ByPathEnd
    )

    foreach ($childRef in @($ParentBot.children)) {
        $childId   = [string]$childRef.id
        $childName = [string]$childRef.name
        $childPath = [string]$childRef.path
        $childBot  = Resolve-ChildBotFromMaps -ChildId $childId -ChildName $childName `
                         -ChildPath $childPath -ById $ById -ByName $ByName -ByPathEnd $ByPathEnd

        $displayPath = if ($childBot) { $childBot.path } else { $childPath }

        [void]$Rows.Add([PSCustomObject]@{
            parent_name  = $ParentBot.name
            parent_path  = $ParentBot.path
            parent_depth = $ParentBot.depth
            child_name   = if ($childName) { $childName } `
                           elseif ($childBot) { $childBot.name } else { $childId }
            child_path   = $displayPath
            child_id     = if ($childId) { $childId } `
                           elseif ($childBot) { $childBot.id } else { "" }
            level        = $Level
            in_scan      = ($null -ne $childBot)
        })

        if ($childBot -and -not $VisitedIds.Contains($childBot.id) -and $Level -lt 15) {
            $newVisited = [System.Collections.Generic.HashSet[string]]::new($VisitedIds)
            [void]$newVisited.Add($childBot.id)
            Recurse-Relationships -ParentBot $childBot -Level ($Level + 1) `
                -VisitedIds $newVisited -Rows $Rows `
                -ById $ById -ByName $ByName -ByPathEnd $ByPathEnd
        }
    }
}


function Build-Relationships {
    param([System.Collections.ArrayList]$Bots)

    $byId      = @{}
    $byName    = @{}
    $byPathEnd = @{}

    foreach ($b in $Bots) {
        $byId[$b.id]     = $b
        $byName[$b.name] = $b
        $end = ($b.path -split '[/\\]')[-1]
        $byPathEnd[$end.ToLower()] = $b
    }

    $rows = [System.Collections.ArrayList]@()

    $allChildIds   = [System.Collections.Generic.HashSet[string]]@()
    $allChildNames = [System.Collections.Generic.HashSet[string]]@()
    foreach ($b in $Bots) {
        foreach ($c in @($b.children)) {
            if ($c.id)   { [void]$allChildIds.Add([string]$c.id) }
            if ($c.name) { [void]$allChildNames.Add([string]$c.name) }
        }
    }

    $masters = [System.Collections.ArrayList]@()
    foreach ($b in $Bots) {
        if (-not $allChildIds.Contains($b.id) -and -not $allChildNames.Contains($b.name)) {
            [void]$masters.Add($b)
        }
    }
    if ($masters.Count -eq 0) {
        foreach ($b in $Bots) { if ($b.children.Count -gt 0) { [void]$masters.Add($b) } }
    }
    if ($masters.Count -eq 0) {
        foreach ($b in $Bots) { [void]$masters.Add($b) }
    }

    foreach ($bot in $masters) {
        $visited = [System.Collections.Generic.HashSet[string]]@()
        [void]$visited.Add($bot.id)
        Recurse-Relationships -ParentBot $bot -Level 1 -VisitedIds $visited `
            -Rows $rows -ById $byId -ByName $byName -ByPathEnd $byPathEnd
    }

    return ,$rows
}


# ─────────────────────────────────────────────────────
#  EXCEL STYLES  (EPPlus helper functions via ImportExcel)
# ─────────────────────────────────────────────────────

$DARK_BLUE  = "1F4E79"
$MID_BLUE   = "2E75B6"
$LIGHT_BLU  = "D6E4F0"
$WHITE_HEX  = "FFFFFF"
$GREEN_BG   = "E8F5E9"
$ORANGE_BG  = "FFF3E0"
$PURPLE_BG  = "EDE7F6"
$RED_DARK   = "8B0000"

$LEVEL_COLORS = @{
    1 = "E3F2FD"
    2 = "E8F5E9"
    3 = "FFF9C4"
    4 = "FCE4EC"
    5 = "F3E5F5"
}


function ConvertTo-DrawingColor {
    param([string]$Hex)
    $Hex = $Hex.TrimStart('#')
    $r   = [Convert]::ToInt32($Hex.Substring(0, 2), 16)
    $g   = [Convert]::ToInt32($Hex.Substring(2, 2), 16)
    $b   = [Convert]::ToInt32($Hex.Substring(4, 2), 16)
    return [System.Drawing.Color]::FromArgb(255, $r, $g, $b)
}


function Set-CellFill {
    param($Cell, [string]$HexColor)
    $Cell.Style.Fill.PatternType = [OfficeOpenXml.Style.ExcelFillStyle]::Solid
    $Cell.Style.Fill.BackgroundColor.SetColor((ConvertTo-DrawingColor $HexColor))
}


function Set-CellBorder {
    param($Cell, [string]$BorderHex = "BFBFBF")
    $s     = [OfficeOpenXml.Style.ExcelBorderStyle]::Thin
    $color = ConvertTo-DrawingColor $BorderHex
    $Cell.Style.Border.Left.Style   = $s
    $Cell.Style.Border.Right.Style  = $s
    $Cell.Style.Border.Top.Style    = $s
    $Cell.Style.Border.Bottom.Style = $s
    $Cell.Style.Border.Left.Color.SetColor($color)
    $Cell.Style.Border.Right.Color.SetColor($color)
    $Cell.Style.Border.Top.Color.SetColor($color)
    $Cell.Style.Border.Bottom.Color.SetColor($color)
}


function Set-CellFont {
    param(
        $Cell,
        [bool]$Bold       = $false,
        [string]$ColorHex = "000000",
        [double]$Size     = 11,
        [string]$Name     = "Arial",
        [bool]$Italic     = $false
    )
    $Cell.Style.Font.Bold   = $Bold
    $Cell.Style.Font.Italic = $Italic
    $Cell.Style.Font.Size   = $Size
    $Cell.Style.Font.Name   = $Name
    $Cell.Style.Font.Color.SetColor((ConvertTo-DrawingColor $ColorHex))
}


function Set-CellAlignment {
    param(
        $Cell,
        [string]$Horizontal = "Left",
        [string]$Vertical   = "Top",
        [bool]$WrapText     = $true
    )
    $Cell.Style.HorizontalAlignment =
        [OfficeOpenXml.Style.ExcelHorizontalAlignment]::$Horizontal
    $Cell.Style.VerticalAlignment =
        [OfficeOpenXml.Style.ExcelVerticalAlignment]::$Vertical
    $Cell.Style.WrapText = $WrapText
}


function Apply-Header {
    param($Ws, [int]$Row, [int]$NCols, [string]$BgHex = "1F4E79")
    for ($c = 1; $c -le $NCols; $c++) {
        $cell = $Ws.Cells[$Row, $c]
        Set-CellFill      -Cell $cell -HexColor $BgHex
        Set-CellFont      -Cell $cell -Bold $true -ColorHex "FFFFFF" -Size 11 -Name "Arial"
        Set-CellAlignment -Cell $cell -Horizontal "Center" -Vertical "Center" -WrapText $true
        Set-CellBorder    -Cell $cell
    }
}


function Set-ColumnWidths {
    param($Ws, [double[]]$Widths)
    for ($i = 0; $i -lt $Widths.Length; $i++) {
        $Ws.Column($i + 1).Width = $Widths[$i]
    }
}


function Get-AltFillHex {
    param([int]$RowIndex, [string]$AltColor = "D6E4F0")
    if ($RowIndex % 2 -eq 0) { return $AltColor } else { return "FFFFFF" }
}


function Write-ExcelRow {
    param(
        $Ws,
        [int]$Row,
        [object[]]$Values,
        [string]$FillHex,
        [bool]$Bold       = $false,
        [string]$FontHex  = "000000",
        [double]$FontSize = 11
    )
    for ($c = 0; $c -lt $Values.Length; $c++) {
        $cell       = $Ws.Cells[$Row, ($c + 1)]
        $cell.Value = $Values[$c]
        Set-CellFill      -Cell $cell -HexColor $FillHex
        Set-CellBorder    -Cell $cell
        Set-CellAlignment -Cell $cell -Horizontal "Left" -Vertical "Top" -WrapText $true
        Set-CellFont      -Cell $cell -Bold $Bold -ColorHex $FontHex -Size $FontSize -Name "Arial"
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKSHEET HELPER
# ═══════════════════════════════════════════════════════════════════════════════
function Get-OrCreateWorksheet {
    param($Pkg, [string]$Name)
    $ws = $Pkg.Workbook.Worksheets[$Name]
    if ($null -eq $ws) {
        $ws = $Pkg.Workbook.Worksheets.Add($Name)
    }
    return $ws
}


# ═══════════════════════════════════════════════════════════════════════════════
#  INCREMENTAL EXCEL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

function Initialize-IncrementalExcel {
    <#
    .SYNOPSIS
        Creates the output workbook with all 9 sheets in the correct order.
        Writes headers + column widths for the 3 incremental sheets now.
        The 5 aggregate sheets and the Scan Issues sheet are left as empty placeholders.
        Sets $script:IncrementalMode = $true.
    #>
    Write-Log "Initializing incremental Excel workbook: $OUTPUT_FILE ..."

    if (-not (Get-Module -ListAvailable -Name ImportExcel)) {
        throw "ImportExcel module not found. Run: Install-Module ImportExcel -Scope CurrentUser"
    }
    if (-not (Get-Module -Name ImportExcel)) {
        Import-Module ImportExcel -ErrorAction Stop
    }
    try {
        [OfficeOpenXml.ExcelPackage]::LicenseContext = [OfficeOpenXml.LicenseContext]::NonCommercial
    } catch {}

    $pkg = New-Object OfficeOpenXml.ExcelPackage

    # ── Sheet 1: Full View (INCREMENTAL -- header written now) ──────────────
    $wsFv = $pkg.Workbook.Worksheets.Add("Full View")
    $wsFv.View.ShowGridLines = $false
    $wsFv.View.FreezePanes(2, 1)
    $fvHdrs = @("Role", "Bot Name", "Bot ID", "Bot Path", "Relationship", "Package Name", "Package Version")
    for ($c = 0; $c -lt $fvHdrs.Count; $c++) { $wsFv.Cells[1, ($c+1)].Value = $fvHdrs[$c] }
    Apply-Header     -Ws $wsFv -Row 1 -NCols $fvHdrs.Count -BgHex $DARK_BLUE
    Set-ColumnWidths -Ws $wsFv -Widths @(22, 32, 12, 60, 38, 28, 20)
    $wsFv.Row(1).Height = 20

    # ── Sheet 2: Summary ─────────────────────────────────────────────────────
    [void]$pkg.Workbook.Worksheets.Add("Summary")

    # ── Sheet 3: Folder Structure (INCREMENTAL -- header now) ────────────────
    $wsFolders = $pkg.Workbook.Worksheets.Add("Folder Structure")
    $wsFolders.View.ShowGridLines = $false
    $wsFolders.View.FreezePanes(2, 1)
    $folderHdrs = @("Folder Name", "Full Path", "Depth Level", "Folder ID", "Parent Folder ID")
    for ($c = 0; $c -lt $folderHdrs.Count; $c++) { $wsFolders.Cells[1, ($c+1)].Value = $folderHdrs[$c] }
    Apply-Header     -Ws $wsFolders -Row 1 -NCols $folderHdrs.Count
    Set-ColumnWidths -Ws $wsFolders -Widths @(35, 60, 14, 15, 18)

    # ── Sheet 4: All Bots (INCREMENTAL -- header now, 14 columns) ────────────
    $wsAllBots = $pkg.Workbook.Worksheets.Add("All Bots")
    $wsAllBots.View.ShowGridLines = $false
    $wsAllBots.View.FreezePanes(2, 1)
    $botHdrs = @(
        "Bot Name", "Full Path", "Depth Level",
        "Has Child Bots?", "# Child Bots", "# Packages",
        "Created By", "Created On", "Last Modified By", "Last Modified On",
        "Locked?", "File Size (bytes)", "Bot ID", "Data Status"
    )
    for ($c = 0; $c -lt $botHdrs.Count; $c++) { $wsAllBots.Cells[1, ($c+1)].Value = $botHdrs[$c] }
    Apply-Header     -Ws $wsAllBots -Row 1 -NCols $botHdrs.Count
    Set-ColumnWidths -Ws $wsAllBots -Widths @(30, 60, 13, 15, 13, 12, 20, 22, 20, 22, 10, 18, 15, 28)

    # ── Sheet 5: Bot Relationships ────────────────────────────────────────────
    [void]$pkg.Workbook.Worksheets.Add("Bot Relationships")

    # ── Sheet 6: Hierarchy Tree ───────────────────────────────────────────────
    [void]$pkg.Workbook.Worksheets.Add("Hierarchy Tree")

    # ── Sheet 7: Packages Per Bot (INCREMENTAL -- header now) ────────────────
    $wsPkgs = $pkg.Workbook.Worksheets.Add("Packages Per Bot")
    $wsPkgs.View.ShowGridLines = $false
    $wsPkgs.View.FreezePanes(2, 1)
    $pkgHdrs = @("Bot Name", "Bot Path", "Package Name", "Package Version")
    for ($c = 0; $c -lt $pkgHdrs.Count; $c++) { $wsPkgs.Cells[1, ($c+1)].Value = $pkgHdrs[$c] }
    Apply-Header     -Ws $wsPkgs -Row 1 -NCols $pkgHdrs.Count
    Set-ColumnWidths -Ws $wsPkgs -Widths @(30, 60, 35, 18)

    # ── Sheet 8: Package Summary ──────────────────────────────────────────────
    [void]$pkg.Workbook.Worksheets.Add("Package Summary")

    # ── Sheet 9: Scan Issues (populated at the end by Add-SheetScanIssues) ───
    [void]$pkg.Workbook.Worksheets.Add("Scan Issues")

    $outPath    = Join-Path (Get-Location).Path $OUTPUT_FILE
    $sheetCount = $pkg.Workbook.Worksheets.Count
    $pkg.SaveAs([System.IO.FileInfo]::new($outPath))
    $pkg.Dispose()
    $pkg = $null

    # Reset all incremental state
    $script:IncrementalMode   = $true
    $script:Excel_BotRow      = 2
    $script:Excel_PkgRow      = 2
    $script:Excel_FolderRow   = 2
    $script:BatchCount        = 0
    $script:BotBatchBuffer.Clear()
    $script:FolderBatchBuffer.Clear()
    $script:BotCache     = @{}
    $script:FullView_Row = 2

    Write-Log "Workbook initialised: $sheetCount sheets, headers written for incremental sheets. File: $outPath"
}


function Flush-IncrementalBatches {
    <#
    .SYNOPSIS
        Appends buffered bot/folder rows to the three incremental sheets and saves.

        SaveAs is retried up to 3 times with 10-second delays.
        Buffers are only cleared on a successful save.
        If all SaveAs attempts fail, the error is logged and buffers are cleared
        to prevent unbounded memory growth during a long run.

        The "All Bots" sheet now has a 14th "Data Status" column.
        Incomplete bots (dataComplete=$false) are highlighted in orange.
    #>
    if ($script:BotBatchBuffer.Count -eq 0 -and $script:FolderBatchBuffer.Count -eq 0) {
        return
    }

    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
    if (-not (Test-Path $outPath)) {
        Write-Log "Flush-IncrementalBatches: output file not found ($outPath). Skipping flush." -Level WARN
        $script:BotBatchBuffer.Clear()
        $script:FolderBatchBuffer.Clear()
        return
    }

    $script:BatchCount++
    $batchNum = $script:BatchCount
    $botCount = $script:BotBatchBuffer.Count
    $fldCount = $script:FolderBatchBuffer.Count
    Write-Log "Flushing batch #$batchNum  ($fldCount folders, $botCount bots) -> $OUTPUT_FILE ..."

    $epkg   = $null
    $saved  = $false

    try {
        if (-not (Get-Module -Name ImportExcel)) {
            Import-Module ImportExcel -ErrorAction Stop
        }
        try {
            [OfficeOpenXml.ExcelPackage]::LicenseContext = [OfficeOpenXml.LicenseContext]::NonCommercial
        } catch {}

        $epkg = New-Object OfficeOpenXml.ExcelPackage ([System.IO.FileInfo]::new($outPath))

        # ── Append to "Folder Structure" ──────────────────────────────────────
        $wsFolders = $epkg.Workbook.Worksheets["Folder Structure"]
        foreach ($f in $script:FolderBatchBuffer) {
            $indent  = ("  " * ($f.depth - 1)) + "F " + $f.name
            $fillHex = Get-AltFillHex -RowIndex $script:Excel_FolderRow
            Write-ExcelRow -Ws $wsFolders -Row $script:Excel_FolderRow `
                           -Values @($indent, $f.path, $f.depth, $f.id, $f.parent_id) `
                           -FillHex $fillHex
            $script:Excel_FolderRow++
        }

        # ── Append to "All Bots" (14 columns) ───────────────────────────────
        $wsAllBots = $epkg.Workbook.Worksheets["All Bots"]
        foreach ($b in $script:BotBatchBuffer) {
            $hasChildren = if ($b.children.Count -gt 0) { "Yes" } else { "No" }
            $dataStatus  = if ($b.dataComplete) { "Complete" } `
                           else { "Partial: " + ($b.dataIssues -join "; ") }

            # Row color priority: orange for incomplete > green for parent > alternating
            $fillHex = if (-not $b.dataComplete) {
                           $ORANGE_BG
                       } elseif ($b.children.Count -gt 0) {
                           $GREEN_BG
                       } else {
                           Get-AltFillHex -RowIndex $script:Excel_BotRow
                       }

            Write-ExcelRow -Ws $wsAllBots -Row $script:Excel_BotRow -Values @(
                $b.name, $b.path, $b.depth,
                $hasChildren, $b.children.Count, $b.packages.Count,
                $b.created_by, $b.created_on,
                $b.modified_by, $b.modified_on,
                $(if ($b.locked) { "Yes" } else { "No" }),
                $b.file_size, $b.id, $dataStatus
            ) -FillHex $fillHex
            $script:Excel_BotRow++
        }

        # ── Append to "Packages Per Bot" ─────────────────────────────────────
        $wsPkgs = $epkg.Workbook.Worksheets["Packages Per Bot"]
        foreach ($b in $script:BotBatchBuffer) {
            if ($b.packages.Count -eq 0) {
                Write-ExcelRow -Ws $wsPkgs -Row $script:Excel_PkgRow `
                    -Values @($b.name, $b.path, "(no packages found)", "") `
                    -FillHex (Get-AltFillHex -RowIndex $script:Excel_PkgRow)
                $script:Excel_PkgRow++
            } else {
                foreach ($pkgItem in @($b.packages)) {
                    Write-ExcelRow -Ws $wsPkgs -Row $script:Excel_PkgRow `
                        -Values @($b.name, $b.path, $pkgItem.name, $pkgItem.version) `
                        -FillHex (Get-AltFillHex -RowIndex $script:Excel_PkgRow)
                    $script:Excel_PkgRow++
                }
            }
        }

        # ── SaveAs with retry (3 attempts, 10 s delay) ───────────────────────
        for ($saveAttempt = 1; $saveAttempt -le 3; $saveAttempt++) {
            try {
                $epkg.SaveAs([System.IO.FileInfo]::new($outPath))
                $saved = $true
                break
            } catch {
                $saveErr = $_.Exception.Message
                Write-Log "Flush-IncrementalBatches: SaveAs attempt $saveAttempt/3 failed: $saveErr" -Level WARN
                if ($saveAttempt -lt 3) {
                    Write-Log "Retrying save in 10 seconds..." -Level WARN
                    Start-Sleep -Seconds 10
                }
            }
        }

        if (-not $saved) {
            Write-Log ("ERROR: All 3 SaveAs attempts failed for batch #$batchNum. " +
                       "Rows written to EPPlus in memory but NOT persisted to disk. " +
                       "If disk is full, free space and check the final report.") -Level ERROR
        } else {
            Write-Log ("Batch #$batchNum flushed. Rows -- AllBots: $($script:Excel_BotRow-1), " +
                       "PkgPerBot: $($script:Excel_PkgRow-1), FolderStructure: $($script:Excel_FolderRow-1)")
        }

    } catch {
        Write-Log "ERROR in Flush-IncrementalBatches (batch #$batchNum): $($_.Exception.Message)" -Level ERROR

    } finally {
        if ($null -ne $epkg) {
            try { $epkg.Dispose() } catch {}
            $epkg = $null
        }
        # Always clear buffers -- on save failure the data is lost for this batch
        # but we MUST clear to prevent unbounded memory growth over a long run.
        $script:BotBatchBuffer.Clear()
        $script:FolderBatchBuffer.Clear()
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: SUMMARY
# ─────────────────────────────────────────────────────
function Add-SheetSummary {
    param($Pkg, $Bots, $Folders, $Relationships)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Summary"
    $ws.View.ShowGridLines = $false
    $ws.Column(1).Width = 3
    $ws.Column(2).Width = 40
    $ws.Column(3).Width = 15

    $title = $ws.Cells[2, 2]
    $title.Value = "Automation Anywhere 360 -- Bot Inventory Report"
    Set-CellFont -Cell $title -Bold $true -Size 16 -ColorHex $DARK_BLUE -Name "Arial"

    $ws.Cells[3, 2].Value = "Generated on $(Get-Date -Format 'dd MMM yyyy HH:mm:ss')"
    Set-CellFont -Cell $ws.Cells[3, 2] -Italic $true -ColorHex "595959" -Size 10 -Name "Arial"

    $ws.Cells[5, 2].Value = "Metric"
    $ws.Cells[5, 3].Value = "Count"
    for ($col = 2; $col -le 3; $col++) {
        Set-CellFill      -Cell $ws.Cells[5, $col] -HexColor $DARK_BLUE
        Set-CellFont      -Cell $ws.Cells[5, $col] -Bold $true -ColorHex "FFFFFF" -Size 11 -Name "Arial"
        Set-CellAlignment -Cell $ws.Cells[5, $col] -Horizontal "Center" -Vertical "Center" -WrapText $true
        Set-CellBorder    -Cell $ws.Cells[5, $col]
    }

    $maxLevel = 0
    foreach ($r in $Relationships) { if ($r.level -gt $maxLevel) { $maxLevel = $r.level } }

    $uniquePkgNames = [System.Collections.Generic.HashSet[string]]@()
    foreach ($b in $Bots) { foreach ($p in @($b.packages)) { [void]$uniquePkgNames.Add($p.name) } }

    $botsWithChildren = 0
    $botsNoChildren   = 0
    $partialBots      = 0
    foreach ($b in $Bots) {
        if ($b.children.Count -gt 0) { $botsWithChildren++ } else { $botsNoChildren++ }
        if (-not $b.dataComplete) { $partialBots++ }
    }

    $stats = @(
        [PSCustomObject]@{ label = "Total Folders Scanned";              value = $Folders.Count }
        [PSCustomObject]@{ label = "Total Bots Found";                   value = $Bots.Count }
        [PSCustomObject]@{ label = "Bots with Child-Bot References";     value = $botsWithChildren }
        [PSCustomObject]@{ label = "Bots with No Children (leaf bots)";  value = $botsNoChildren }
        [PSCustomObject]@{ label = "Total Parent -> Child Links";        value = $Relationships.Count }
        [PSCustomObject]@{ label = "Max Nesting Depth (levels)";         value = $maxLevel }
        [PSCustomObject]@{ label = "Total Unique Packages Found";        value = $uniquePkgNames.Count }
        [PSCustomObject]@{ label = "Bots with Incomplete Data (check Scan Issues sheet)"; value = $partialBots }
        [PSCustomObject]@{ label = "Failed Folder Scans";                value = $script:FailedFolders.Count }
        [PSCustomObject]@{ label = "Failed API Calls (total)";           value = $script:FailedApiCalls.Count }
    )

    for ($i = 0; $i -lt $stats.Count; $i++) {
        $rowIdx  = $i + 6
        $fillHex = Get-AltFillHex -RowIndex $rowIdx

        $labelCell = $ws.Cells[$rowIdx, 2]
        $labelCell.Value = $stats[$i].label
        Set-CellFill   -Cell $labelCell -HexColor $fillHex
        Set-CellBorder -Cell $labelCell
        Set-CellFont   -Cell $labelCell -Name "Arial" -Size 11

        $valCell = $ws.Cells[$rowIdx, 3]
        $valCell.Value = $stats[$i].value
        Set-CellFill      -Cell $valCell -HexColor $fillHex
        Set-CellBorder    -Cell $valCell
        Set-CellFont      -Cell $valCell -Bold $true -Name "Arial" -Size 11
        Set-CellAlignment -Cell $valCell -Horizontal "Center" -Vertical "Center" -WrapText $true
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: FOLDER STRUCTURE
# ─────────────────────────────────────────────────────
function Add-SheetFolders {
    param($Pkg, $Folders)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Folder Structure"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $headers = @("Folder Name", "Full Path", "Depth Level", "Folder ID", "Parent Folder ID")
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count
    Set-ColumnWidths -Ws $ws -Widths @(35, 60, 14, 15, 18)

    $i = 2
    foreach ($f in $Folders) {
        $indent  = ("  " * ($f.depth - 1)) + "F " + $f.name
        $fillHex = Get-AltFillHex -RowIndex $i
        Write-ExcelRow -Ws $ws -Row $i -Values @($indent, $f.path, $f.depth, $f.id, $f.parent_id) `
                       -FillHex $fillHex
        $i++
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: ALL BOTS  (14 columns, 14th = Data Status)
# ─────────────────────────────────────────────────────
function Add-SheetAllBots {
    param($Pkg, $Bots)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "All Bots"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $headers = @(
        "Bot Name", "Full Path", "Depth Level",
        "Has Child Bots?", "# Child Bots", "# Packages",
        "Created By", "Created On", "Last Modified By", "Last Modified On",
        "Locked?", "File Size (bytes)", "Bot ID", "Data Status"
    )
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count
    Set-ColumnWidths -Ws $ws -Widths @(30, 60, 13, 15, 13, 12, 20, 22, 20, 22, 10, 18, 15, 28)

    $i = 2
    foreach ($b in $Bots) {
        $hasChildren = if ($b.children.Count -gt 0) { "Yes" } else { "No" }
        $dataStatus  = if ($b.dataComplete) { "Complete" } `
                       else { "Partial: " + ($b.dataIssues -join "; ") }

        # Row color: orange for incomplete > green for parent bot > alternating
        $fillHex = if (-not $b.dataComplete) {
                       $ORANGE_BG
                   } elseif ($b.children.Count -gt 0) {
                       $GREEN_BG
                   } else {
                       Get-AltFillHex -RowIndex $i
                   }

        Write-ExcelRow -Ws $ws -Row $i -Values @(
            $b.name, $b.path, $b.depth,
            $hasChildren, $b.children.Count, $b.packages.Count,
            $b.created_by, $b.created_on,
            $b.modified_by, $b.modified_on,
            $(if ($b.locked) { "Yes" } else { "No" }),
            $b.file_size, $b.id, $dataStatus
        ) -FillHex $fillHex
        $i++
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: BOT RELATIONSHIPS
# ─────────────────────────────────────────────────────
function Add-SheetRelationships {
    param($Pkg, $Relationships)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Bot Relationships"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $headers = @(
        "Master / Parent Bot", "Parent Bot Path",
        "Child Bot Name",      "Child Bot Path",
        "Relationship Level",  "Level Label",
        "Child Found in Scan?"
    )
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count
    Set-ColumnWidths -Ws $ws -Widths @(30, 55, 30, 55, 18, 22, 20)

    $labels = @{
        1 = "Direct Child"
        2 = "Grandchild"
        3 = "Great-Grandchild"
        4 = "Level 4"
        5 = "Level 5+"
    }

    $i = 2
    foreach ($r in $Relationships) {
        $lvl        = $r.level
        $colorHex   = if ($LEVEL_COLORS.ContainsKey($lvl)) { $LEVEL_COLORS[$lvl] } else { "F5F5F5" }
        $levelLabel = if ($labels.ContainsKey($lvl)) { $labels[$lvl] } else { "Level $lvl" }
        $inScanStr  = if ($r.in_scan) { "Yes" } else { "Not in scanned folder" }

        Write-ExcelRow -Ws $ws -Row $i -Values @(
            $r.parent_name, $r.parent_path,
            $r.child_name,  $r.child_path,
            $lvl, $levelLabel, $inScanStr
        ) -FillHex $colorHex
        $i++
    }

    $legendRow = $Relationships.Count + 4
    $ws.Cells[$legendRow, 1].Value = "Legend:"
    Set-CellFont -Cell $ws.Cells[$legendRow, 1] -Bold $true -Name "Arial"

    $legendItems = @(
        [PSCustomObject]@{ color = $LEVEL_COLORS[1]; label = "Direct Child (Level 1)" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[2]; label = "Grandchild (Level 2)" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[3]; label = "Great-Grandchild (Level 3)" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[4]; label = "Level 4" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[5]; label = "Level 5+" }
    )
    for ($off = 0; $off -lt $legendItems.Count; $off++) {
        $cell       = $ws.Cells[($legendRow + $off + 1), 1]
        $cell.Value = $legendItems[$off].label
        Set-CellFill   -Cell $cell -HexColor $legendItems[$off].color
        Set-CellBorder -Cell $cell
        Set-CellFont   -Cell $cell -Name "Arial" -Size 10
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: HIERARCHY TREE
# ─────────────────────────────────────────────────────
function Draw-BotTreeNode {
    param(
        $Bot,
        [int]$Level,
        [System.Collections.Generic.HashSet[string]]$Visited,
        $Ws,
        [hashtable]$ById,
        [hashtable]$ByName
    )

    $prefix   = ("    " * $Level) + $(if ($Level -eq 0) { "Bot " } else { "-> " })
    $colorHex = if ($LEVEL_COLORS.ContainsKey($Level)) { $LEVEL_COLORS[$Level] } else { "FAFAFA" }

    Write-ExcelRow -Ws $Ws -Row $script:TreeRow -Values @(
        "$prefix$($Bot.name)", $Bot.path, $Bot.id, $Bot.packages.Count
    ) -FillHex $colorHex
    $script:TreeRow++

    foreach ($childRaw in @($Bot.children)) {
        $childId   = [string]$(if ($childRaw.id)       { $childRaw.id }       `
                               elseif ($childRaw.fileId) { $childRaw.fileId } else { "" })
        $childName = [string]$(if ($childRaw.name)       { $childRaw.name }       `
                               elseif ($childRaw.fileName) { $childRaw.fileName } else { "" })
        $childBot  = if ($childId -and $ById.ContainsKey($childId))           { $ById[$childId]    } `
                     elseif ($childName -and $ByName.ContainsKey($childName)) { $ByName[$childName] } `
                     else { $null }

        if ($childBot -and -not $Visited.Contains($childId)) {
            $newVisited = [System.Collections.Generic.HashSet[string]]::new($Visited)
            [void]$newVisited.Add($childId)
            Draw-BotTreeNode -Bot $childBot -Level ($Level + 1) -Visited $newVisited `
                             -Ws $Ws -ById $ById -ByName $ByName
        } elseif (-not $childBot) {
            Write-ExcelRow -Ws $Ws -Row $script:TreeRow -Values @(
                ("    " * ($Level + 1)) + "-> [OUTSIDE FOLDER] $childName",
                [string]$(if ($childRaw.path) { $childRaw.path } else { "" }),
                $childId, ""
            ) -FillHex "FFF9C4"
            $script:TreeRow++
        }
    }
}


function Add-SheetTree {
    param($Pkg, $Bots)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Hierarchy Tree"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $ws.Cells[1, 1].Value = "Bot Hierarchy (indented tree view)"
    $ws.Cells[1, 2].Value = "Full Path"
    $ws.Cells[1, 3].Value = "Bot ID"
    $ws.Cells[1, 4].Value = "# Packages"
    Apply-Header     -Ws $ws -Row 1 -NCols 4
    Set-ColumnWidths -Ws $ws -Widths @(55, 65, 15, 13)

    $byId   = @{}
    $byName = @{}
    foreach ($b in $Bots) {
        $byId[$b.id]     = $b
        $byName[$b.name] = $b
    }

    $allChildIds_t = [System.Collections.Generic.HashSet[string]]@()
    foreach ($b in $Bots) {
        foreach ($c in @($b.children)) {
            $cid = if ($c.id) { [string]$c.id } elseif ($c.fileId) { [string]$c.fileId } else { "" }
            if ($cid) { [void]$allChildIds_t.Add($cid) }
        }
    }

    $masters_t = [System.Collections.ArrayList]@()
    foreach ($b in $Bots) {
        if (-not $allChildIds_t.Contains($b.id)) { [void]$masters_t.Add($b) }
    }

    $script:TreeRow = 2
    foreach ($bot in $masters_t) {
        Draw-BotTreeNode -Bot $bot -Level 0 `
            -Visited ([System.Collections.Generic.HashSet[string]]@($bot.id)) `
            -Ws $ws -ById $byId -ByName $byName
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: PACKAGES PER BOT
# ─────────────────────────────────────────────────────
function Add-SheetPackagesPerBot {
    param($Pkg, $Bots)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Packages Per Bot"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $headers = @("Bot Name", "Bot Path", "Package Name", "Package Version")
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count
    Set-ColumnWidths -Ws $ws -Widths @(30, 60, 35, 18)

    $i = 2
    foreach ($bot in $Bots) {
        if ($bot.packages.Count -eq 0) {
            Write-ExcelRow -Ws $ws -Row $i `
                -Values @($bot.name, $bot.path, "(no packages found)", "") `
                -FillHex (Get-AltFillHex -RowIndex $i)
            $i++
        } else {
            foreach ($pkgItem in @($bot.packages)) {
                Write-ExcelRow -Ws $ws -Row $i `
                    -Values @($bot.name, $bot.path, $pkgItem.name, $pkgItem.version) `
                    -FillHex (Get-AltFillHex -RowIndex $i)
                $i++
            }
        }
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: PACKAGE SUMMARY
# ─────────────────────────────────────────────────────
function Add-SheetPackageSummary {
    param($Pkg, $Bots)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Package Summary"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $usage = @{}
    foreach ($bot in $Bots) {
        foreach ($p in @($bot.packages)) {
            $key = "$($p.name)|$($p.version)"
            if (-not $usage.ContainsKey($key)) {
                $usage[$key] = [System.Collections.Generic.HashSet[string]]@()
            }
            [void]$usage[$key].Add($bot.name)
        }
    }

    $headers = @("Package Name", "Version", "# Bots Using It", "Bots Using This Package")
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count
    Set-ColumnWidths -Ws $ws -Widths @(35, 18, 15, 70)

    $sortedKeys = $usage.Keys | Sort-Object `
        @{ Expression = { ($_ -split '\|', 2)[0].ToLower() } }, `
        @{ Expression = { ($_ -split '\|', 2)[1] } }

    $i = 2
    foreach ($key in $sortedKeys) {
        $parts    = $key -split '\|', 2
        $pkgName  = $parts[0]
        $pkgVer   = if ($parts.Length -gt 1) { $parts[1] } else { "" }
        $botNames = $usage[$key]
        $botsList = ($botNames | Sort-Object) -join ", "
        Write-ExcelRow -Ws $ws -Row $i `
            -Values @($pkgName, $pkgVer, $botNames.Count, $botsList) `
            -FillHex (Get-AltFillHex -RowIndex $i)
        $i++
    }
}


# ─────────────────────────────────────────────────────
#  SHEET: SCAN ISSUES  (9th sheet -- new)
# ─────────────────────────────────────────────────────
function Add-SheetScanIssues {
    <#
    .SYNOPSIS
        Writes the "Scan Issues" sheet.

        Sources:
          $script:FailedApiCalls  -- GetContent / GetDependencies failures per bot
          $script:FailedFolders   -- ListFolderContents failures per folder

        If no issues were detected, writes a single "All clear" row in green.

        Columns:
          Issue Type | Bot/Folder Name | Path | ID | Failed API | HTTP Status |
          Error Message | Impact | Timestamp
    #>
    param($Pkg)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Scan Issues"
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $headers = @(
        "Issue Type", "Bot/Folder Name or ID", "Path",
        "ID", "Failed API Endpoint", "HTTP Status",
        "Error Message", "Data Impact", "Timestamp"
    )
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count -BgHex $RED_DARK
    Set-ColumnWidths -Ws $ws -Widths @(22, 22, 50, 15, 22, 12, 60, 40, 22)

    $i = 2

    # ── Bot-level API failures ────────────────────────────────────────────────
    foreach ($issue in $script:FailedApiCalls) {
        $issueType = switch ($issue.apiType) {
            "GetContent"      { "Bot Content API Failed" }
            "GetDependencies" { "Bot Dependencies API Failed" }
            default           { "API Failure" }
        }
        $impact = switch ($issue.apiType) {
            "GetContent"      { "Package list may be incomplete; child bot refs from content parse are missing" }
            "GetDependencies" { "Child bot relationship list may be incomplete" }
            default           { "Data for this bot may be incomplete" }
        }
        $statusDisplay = if ($issue.statusCode -gt 0) { [string]$issue.statusCode } `
                         else { "Network / Timeout" }

        Write-ExcelRow -Ws $ws -Row $i -Values @(
            $issueType,
            $issue.fileId,
            "",                 # path not available at API call time; correlate via ID in All Bots
            $issue.fileId,
            $issue.apiType,
            $statusDisplay,
            $issue.error,
            $impact,
            $issue.timestamp
        ) -FillHex $ORANGE_BG
        $i++
    }

    # ── Folder scan failures ──────────────────────────────────────────────────
    foreach ($ff in $script:FailedFolders) {
        $statusDisplay = if ($ff.statusCode -gt 0) { [string]$ff.statusCode } `
                         else { "Network / Timeout" }

        Write-ExcelRow -Ws $ws -Row $i -Values @(
            "Folder Scan Failed",
            $ff.folderId,
            $ff.folderPath,
            $ff.folderId,
            "POST /v2/repository/folders/{id}/list",
            $statusDisplay,
            $ff.error,
            "All bots and sub-folders within this folder may be MISSING from the report",
            $ff.timestamp
        ) -FillHex "FCE4EC"   # light pink -- folder failures are higher impact
        $i++
    }

    # ── All clear message ─────────────────────────────────────────────────────
    if ($i -eq 2) {
        $cell       = $ws.Cells[2, 1]
        $cell.Value = "No scan issues detected -- all API calls succeeded and all folders were scanned successfully."
        Set-CellFont      -Cell $cell -ColorHex "2E7D32" -Bold $true -Name "Arial" -Size 11
        Set-CellFill      -Cell $cell -HexColor "E8F5E9"
        Set-CellBorder    -Cell $cell
        Set-CellAlignment -Cell $cell -Horizontal "Left" -Vertical "Center" -WrapText $true
    }

    Write-Log "Scan Issues sheet: $($script:FailedApiCalls.Count) API failure(s), $($script:FailedFolders.Count) folder failure(s) recorded."
}


# ─────────────────────────────────────────────────────
#  SHEET: FULL VIEW  (Bot + Relationships + Packages)
# ─────────────────────────────────────────────────────

$ROLE_LABELS = @{
    0 = "MASTER"
    1 = "  +- Child (L1)"
    2 = "  |  +- Child (L2)"
    3 = "  |  |  +- Child (L3)"
    4 = "  |  |  |  +- Child (L4)"
}


function Write-BotBlock {
    param(
        $Bot,
        [int]$Level,
        [string]$RelLabel,
        [System.Collections.Generic.HashSet[string]]$VisitedIds,
        $Ws,
        [hashtable]$ById,
        [hashtable]$ByName,
        [hashtable]$ByPathEnd
    )

    $roleLabel = if ($ROLE_LABELS.ContainsKey($Level)) { $ROLE_LABELS[$Level] } `
                 else { "  L$Level Child" }
    $packages  = @($Bot.packages)

    if ($Level -eq 0) {
        $botFillHex  = $DARK_BLUE
        $botFontHex  = "FFFFFF"
        $botFontBold = $true
        $botFontSize = 11
    } else {
        $botFillHex  = if ($LEVEL_COLORS.ContainsKey($Level)) { $LEVEL_COLORS[$Level] } else { "EEF2F7" }
        $botFontHex  = switch ($Level) {
            1       { "0D3B66" }
            2       { "1A5276" }
            3       { "1F618D" }
            default { "000000" }
        }
        $botFontBold = ($Level -eq 1)
        $botFontSize = 10
    }

    $firstPkg = if ($packages.Count -gt 0) { $packages[0] } `
                else { [PSCustomObject]@{ name = "(none)"; version = "" } }

    $rowVals = @(
        $roleLabel, $Bot.name, $Bot.id, $Bot.path,
        $RelLabel, $firstPkg.name, $firstPkg.version
    )
    for ($c = 0; $c -lt $rowVals.Count; $c++) {
        $cell       = $Ws.Cells[$script:FvRow, ($c + 1)]
        $cell.Value = $rowVals[$c]
        Set-CellFill      -Cell $cell -HexColor $botFillHex
        Set-CellBorder    -Cell $cell
        Set-CellAlignment -Cell $cell -Horizontal "Left" -Vertical "Top" -WrapText $true
        if ($c -lt 5) {
            Set-CellFont -Cell $cell -Bold $botFontBold -ColorHex $botFontHex `
                         -Size $botFontSize -Name "Arial"
        } else {
            Set-CellFont -Cell $cell -ColorHex "444444" -Size 10 -Name "Arial"
        }
    }
    $Ws.Row($script:FvRow).Height = 16
    $script:FvRow++

    for ($pkgIdx = 1; $pkgIdx -lt $packages.Count; $pkgIdx++) {
        $pkgItem = $packages[$pkgIdx]
        for ($c = 1; $c -le 7; $c++) {
            $cell = $Ws.Cells[$script:FvRow, $c]
            Set-CellFill      -Cell $cell -HexColor "F8F9FA"
            Set-CellBorder    -Cell $cell
            Set-CellAlignment -Cell $cell -Horizontal "Left" -Vertical "Top" -WrapText $true
        }
        $Ws.Cells[$script:FvRow, 6].Value = $pkgItem.name
        $Ws.Cells[$script:FvRow, 7].Value = $pkgItem.version
        Set-CellFont -Cell $Ws.Cells[$script:FvRow, 6] -ColorHex "444444" -Size 10 -Name "Arial"
        Set-CellFont -Cell $Ws.Cells[$script:FvRow, 7] -ColorHex "444444" -Size 10 -Name "Arial"
        Set-CellFill -Cell $Ws.Cells[$script:FvRow, 6] -HexColor "F8F9FA"
        Set-CellFill -Cell $Ws.Cells[$script:FvRow, 7] -HexColor "F8F9FA"
        $Ws.Row($script:FvRow).Height = 15
        $script:FvRow++
    }

    foreach ($childRef in @($Bot.children)) {
        $childId   = [string]$childRef.id
        $childName = [string]$childRef.name
        $childPath = [string]$childRef.path
        $childBot  = Resolve-ChildBotFromMaps -ChildId $childId -ChildName $childName `
                         -ChildPath $childPath -ById $ById -ByName $ByName -ByPathEnd $ByPathEnd

        $relLabel2 = "Child of $($Bot.name)"

        if ($childBot -and -not $VisitedIds.Contains($childBot.id)) {
            $newVis = [System.Collections.Generic.HashSet[string]]::new($VisitedIds)
            [void]$newVis.Add($childBot.id)
            Write-BotBlock -Bot $childBot -Level ($Level + 1) -RelLabel $relLabel2 `
                -VisitedIds $newVis -Ws $Ws -ById $ById -ByName $ByName -ByPathEnd $ByPathEnd
        } else {
            for ($c = 1; $c -le 7; $c++) {
                $cell = $Ws.Cells[$script:FvRow, $c]
                Set-CellFill   -Cell $cell -HexColor "FFF9C4"
                Set-CellBorder -Cell $cell
            }
            $phRole = if ($ROLE_LABELS.ContainsKey($Level+1)) { $ROLE_LABELS[$Level+1] } `
                      else { "  L$($Level+1) Child" }
            $Ws.Cells[$script:FvRow, 1].Value = $phRole
            $Ws.Cells[$script:FvRow, 2].Value = if ($childName) { $childName } else { $childId }
            $Ws.Cells[$script:FvRow, 3].Value = $childId
            $Ws.Cells[$script:FvRow, 4].Value = $childPath
            $Ws.Cells[$script:FvRow, 5].Value = "Child of $($Bot.name)"
            $Ws.Cells[$script:FvRow, 6].Value = "[OUTSIDE SCANNED FOLDER]"
            $script:FvRow++
        }
    }
}


function Add-SheetFullView {
    param($Pkg, $Bots)

    $ws = Get-OrCreateWorksheet -Pkg $Pkg -Name "Full View"
    # Clear incremental data written during the walk -- this final write is authoritative
    # and applies correct MASTER/Child role labels using the complete bot graph.
    $ws.Cells.Clear()
    $ws.View.ShowGridLines = $false
    $ws.View.FreezePanes(2, 1)

    $headers = @("Role", "Bot Name", "Bot ID", "Bot Path",
                 "Relationship", "Package Name", "Package Version")
    for ($c = 0; $c -lt $headers.Count; $c++) { $ws.Cells[1, ($c+1)].Value = $headers[$c] }
    Apply-Header     -Ws $ws -Row 1 -NCols $headers.Count -BgHex $DARK_BLUE
    Set-ColumnWidths -Ws $ws -Widths @(22, 32, 12, 60, 38, 28, 20)
    $ws.Row(1).Height = 20

    $byId_fv      = @{}
    $byName_fv    = @{}
    $byPathEnd_fv = @{}
    foreach ($b in $Bots) {
        $byId_fv[$b.id]     = $b
        $byName_fv[$b.name] = $b
        $end = ($b.path -split '[/\\]')[-1]
        $byPathEnd_fv[$end.ToLower()] = $b
    }

    $allChildIds_fv   = [System.Collections.Generic.HashSet[string]]@()
    $allChildNames_fv = [System.Collections.Generic.HashSet[string]]@()
    foreach ($b in $Bots) {
        foreach ($c in @($b.children)) {
            if ($c.id)   { [void]$allChildIds_fv.Add([string]$c.id) }
            if ($c.name) { [void]$allChildNames_fv.Add([string]$c.name) }
        }
    }

    $masters_fv = [System.Collections.ArrayList]@()
    foreach ($b in $Bots) {
        if (-not $allChildIds_fv.Contains($b.id) -and
            -not $allChildNames_fv.Contains($b.name)) {
            [void]$masters_fv.Add($b)
        }
    }
    if ($masters_fv.Count -eq 0) {
        foreach ($b in $Bots) { if ($b.children.Count -gt 0) { [void]$masters_fv.Add($b) } }
    }
    if ($masters_fv.Count -eq 0) {
        foreach ($b in $Bots) { [void]$masters_fv.Add($b) }
    }

    $script:FvRow = 2
    for ($mi = 0; $mi -lt $masters_fv.Count; $mi++) {
        $master = $masters_fv[$mi]
        Write-BotBlock -Bot $master -Level 0 -RelLabel "MASTER" `
            -VisitedIds ([System.Collections.Generic.HashSet[string]]@($master.id)) `
            -Ws $ws -ById $byId_fv -ByName $byName_fv -ByPathEnd $byPathEnd_fv

        if ($mi -lt ($masters_fv.Count - 1)) {
            for ($c = 1; $c -le 7; $c++) {
                Set-CellFill   -Cell $ws.Cells[$script:FvRow, $c] -HexColor "E8EDF2"
                Set-CellBorder -Cell $ws.Cells[$script:FvRow, $c]
            }
            $ws.Row($script:FvRow).Height = 8
            $script:FvRow++
        }
    }

    $legendRow = $script:FvRow + 2
    $ws.Cells[$legendRow, 1].Value = "Legend"
    Set-CellFont -Cell $ws.Cells[$legendRow, 1] -Bold $true -Name "Arial" -Size 10

    $legendItems = @(
        [PSCustomObject]@{ color = $DARK_BLUE;       label = "MASTER bot";                             fontHex = "FFFFFF" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[1]; label = "Direct Child (Level 1)";                 fontHex = "000000" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[2]; label = "Grandchild (Level 2)";                   fontHex = "000000" }
        [PSCustomObject]@{ color = $LEVEL_COLORS[3]; label = "Great-Grandchild (Level 3)";             fontHex = "000000" }
        [PSCustomObject]@{ color = "FFF9C4";          label = "Referenced but outside scanned folder";  fontHex = "000000" }
        [PSCustomObject]@{ color = $ORANGE_BG;        label = "Incomplete data (see Scan Issues sheet)"; fontHex = "000000" }
    )
    for ($off = 0; $off -lt $legendItems.Count; $off++) {
        $cell       = $ws.Cells[($legendRow + $off + 1), 1]
        $cell.Value = $legendItems[$off].label
        Set-CellFill   -Cell $cell -HexColor $legendItems[$off].color
        Set-CellBorder -Cell $cell
        Set-CellFont   -Cell $cell -Name "Arial" -Size 10 -ColorHex $legendItems[$off].fontHex
        $ws.Cells[($legendRow + $off + 1), 2].Value = ""
    }
}


# ─────────────────────────────────────────────────────
#  ASSEMBLE REPORT
# ─────────────────────────────────────────────────────
function Invoke-GenerateReport {
    param($Bots, $Folders, $Relationships)

    if (-not (Get-Module -ListAvailable -Name ImportExcel)) {
        throw "ImportExcel module not found. Install it with: Install-Module ImportExcel -Scope CurrentUser"
    }
    if (-not (Get-Module -Name ImportExcel)) {
        Import-Module ImportExcel -ErrorAction Stop
    }

    try {
        [OfficeOpenXml.ExcelPackage]::LicenseContext = [OfficeOpenXml.LicenseContext]::NonCommercial
    } catch {}

    $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE

    # ── Excel row-limit guard ─────────────────────────────────────────────────
    $maxExcelRows = 1048576
    $pkgRowEstimate = 0
    foreach ($b in $Bots) { $pkgRowEstimate += [Math]::Max(1, $b.packages.Count) }

    if ($Bots.Count -gt ($maxExcelRows - 10000)) {
        Write-Log ("WARNING: Bot count ($($Bots.Count)) is approaching the Excel row limit (1,048,576). " +
                   "The 'All Bots' sheet may be truncated.") -Level WARN
    }
    if ($pkgRowEstimate -gt ($maxExcelRows - 10000)) {
        Write-Log ("WARNING: Estimated package rows ($pkgRowEstimate) approaching Excel row limit. " +
                   "The 'Packages Per Bot' sheet may be truncated.") -Level WARN
    }
    if ($Relationships.Count -gt ($maxExcelRows - 10000)) {
        Write-Log ("WARNING: Relationship rows ($($Relationships.Count)) approaching Excel row limit. " +
                   "The 'Bot Relationships' sheet may be truncated.") -Level WARN
    }

    # ── Open existing file (incremental) or create new (legacy) ──────────────
    if ($script:IncrementalMode -and (Test-Path $outPath)) {
        Write-Log "Opening existing incremental workbook to add aggregate sheets: $outPath"
        $pkg = New-Object OfficeOpenXml.ExcelPackage ([System.IO.FileInfo]::new($outPath))
    } else {
        Write-Log "Creating new workbook (legacy / non-incremental mode)."
        $pkg = New-Object OfficeOpenXml.ExcelPackage
    }

    # ── Aggregate sheets (always written at end) ──────────────────────────────
    Write-Log "Building sheet: Full View ..."
    Add-SheetFullView      -Pkg $pkg -Bots $Bots

    Write-Log "Building sheet: Summary ..."
    Add-SheetSummary       -Pkg $pkg -Bots $Bots -Folders $Folders -Relationships $Relationships

    # Incremental sheets: only write in legacy (non-incremental) mode
    if (-not $script:IncrementalMode) {
        Write-Log "Building sheet: Folder Structure ..."
        Add-SheetFolders       -Pkg $pkg -Folders $Folders

        Write-Log "Building sheet: All Bots ..."
        Add-SheetAllBots       -Pkg $pkg -Bots $Bots
    }

    Write-Log "Building sheet: Bot Relationships ..."
    Add-SheetRelationships -Pkg $pkg -Relationships $Relationships

    Write-Log "Building sheet: Hierarchy Tree ..."
    Add-SheetTree          -Pkg $pkg -Bots $Bots

    if (-not $script:IncrementalMode) {
        Write-Log "Building sheet: Packages Per Bot ..."
        Add-SheetPackagesPerBot -Pkg $pkg -Bots $Bots
    }

    Write-Log "Building sheet: Package Summary ..."
    Add-SheetPackageSummary -Pkg $pkg -Bots $Bots

    Write-Log "Building sheet: Scan Issues ..."
    Add-SheetScanIssues     -Pkg $pkg

    $pkg.SaveAs([System.IO.FileInfo]::new($outPath))
    $pkg.Dispose()

    Write-Log "Report saved -> $outPath"
}


# ─────────────────────────────────────────────────────
#  RUN SUMMARY  (final console output)
# ─────────────────────────────────────────────────────
function Write-RunSummary {
    param(
        [System.Collections.ArrayList]$Bots,
        [System.Collections.ArrayList]$Folders
    )

    $totalBots    = $Bots.Count
    $completeBots = 0
    $partialBots  = 0
    foreach ($b in $Bots) {
        if ($b.dataComplete) { $completeBots++ } else { $partialBots++ }
    }
    $successRate = if ($totalBots -gt 0) {
        [Math]::Round(($completeBots / $totalBots) * 100, 1)
    } else {
        100.0
    }

    $failedFolderCount = $script:FailedFolders.Count
    $failedApiCount    = $script:FailedApiCalls.Count

    $rateColor    = if ($successRate -lt 95) { "Yellow" } else { "Green" }
    $partialColor = if ($partialBots  -gt 0) { "Yellow" } else { "Green" }
    $folderColor  = if ($failedFolderCount -gt 0) { "Red" } else { "Green" }
    $apiColor     = if ($failedApiCount    -gt 0) { "Yellow" } else { "Green" }

    Write-Host ""
    Write-Host "  ═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "    SCAN COMPLETE -- RUN SUMMARY" -ForegroundColor Cyan
    Write-Host "  ═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ("    Total Folders Scanned  : " + $Folders.Count) -ForegroundColor White
    Write-Host ("    Total Bots Found       : " + $totalBots) -ForegroundColor White
    Write-Host ("    Complete Bot Records   : " + $completeBots) -ForegroundColor Green
    Write-Host ("    Partial Bot Records    : " + $partialBots) -ForegroundColor $partialColor
    Write-Host ("    Data Completeness Rate : ${successRate}%") -ForegroundColor $rateColor
    Write-Host ("    Failed Folder Scans    : " + $failedFolderCount) -ForegroundColor $folderColor
    Write-Host ("    Failed API Calls Total : " + $failedApiCount) -ForegroundColor $apiColor

    if ($partialBots -gt 0 -or $failedFolderCount -gt 0) {
        Write-Host ""
        Write-Host "  *** ATTENTION: Data quality issues detected ***" -ForegroundColor Yellow
        Write-Host "    > Check the 'Scan Issues' sheet in $OUTPUT_FILE for full details." -ForegroundColor Yellow
        Write-Host "    > Orange rows in the 'All Bots' sheet indicate incomplete bot data." -ForegroundColor Yellow
        if ($failedFolderCount -gt 0) {
            Write-Host "    > $failedFolderCount folder(s) could not be listed -- bots in those folders are MISSING." -ForegroundColor Red
        }
    } else {
        Write-Host ""
        Write-Host "    All API calls succeeded. Report data is complete." -ForegroundColor Green
    }

    Write-Host "  ═══════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT / RESUME
#  Allows a crashed or interrupted run to continue from its last save-point
#  rather than starting the entire folder walk from scratch.
#
#  Save point: after every successful Flush-IncrementalBatches (i.e. every
#  BATCH_SIZE bots).  The checkpoint JSON is written atomically (temp-rename).
#
#  Resume: at startup, if the checkpoint file exists the user is asked Y/N.
#    Y  -> restore $bots, $folders, row-counters, failure tracking, completed
#           folder set; skip the Excel init; continue the walk.
#    N  -> delete checkpoint, proceed as a fresh run.
#
#  Guarantees:
#    - No duplicate API calls: BotCache is rebuilt from restored bots.
#    - No duplicate Excel rows: RestoredBotIds / RestoredFolderIds guards.
#    - Completed folders (all items processed) are skipped entirely.
#    - Partially-done folders are re-scanned; deduplication prevents doubling.
#    - Final aggregate sheets (Bot Tree, Summary, etc.) are always complete
#      because Invoke-GenerateReport uses the full in-memory $bots list.
#    - Checkpoint is deleted after a successful end-to-end run.
# ═══════════════════════════════════════════════════════════════════════════════

function Save-Checkpoint {
    <#
    .SYNOPSIS
        Serialises current run state to AA_Bot_Report_checkpoint.json.
        Uses an atomic write (temp file + rename) to prevent a partial checkpoint
        from being read on the next startup.
    #>
    param(
        [System.Collections.ArrayList]$Bots,
        [System.Collections.ArrayList]$Folders
    )
    try {
        # Serialise bot records – ConvertTo-Json handles nested PSCustomObjects.
        # Depth 8 covers children/packages/dataIssues without risk of recursion.
        $botArray = @($Bots | ForEach-Object {
            [PSCustomObject]@{
                id           = $_.id
                name         = $_.name
                path         = $_.path
                parent_id    = $_.parent_id
                depth        = $_.depth
                children     = @($_.children | ForEach-Object { [PSCustomObject]@{ id=$_.id; name=$_.name; path=$_.path } })
                packages     = @($_.packages | ForEach-Object { [PSCustomObject]@{ name=$_.name; version=$_.version } })
                created_by   = $_.created_by
                created_on   = $_.created_on
                modified_by  = $_.modified_by
                modified_on  = $_.modified_on
                locked       = $_.locked
                file_size    = $_.file_size
                dataComplete = $_.dataComplete
                dataIssues   = @($_.dataIssues)
            }
        })

        $folderArray = @($Folders | ForEach-Object {
            [PSCustomObject]@{
                id        = $_.id
                name      = $_.name
                path      = $_.path
                parent_id = $_.parent_id
                depth     = $_.depth
            }
        })

        $cpData = [ordered]@{
            version          = "2.0"
            savedAt          = (Get-Date).ToString("o")
            botCount         = $Bots.Count
            folderCount      = $Folders.Count
            completedFolders = @($script:CompletedFolders)
            rowCounters      = [ordered]@{
                botRow        = $script:Excel_BotRow
                pkgRow        = $script:Excel_PkgRow
                folderRow     = $script:Excel_FolderRow
                fullViewRow   = $script:FullView_Row
                batchCount    = $script:BatchCount
                botCounter    = $script:BotCounter
                folderCounter = $script:FolderCounter
            }
            failedApiCalls   = @($script:FailedApiCalls | ForEach-Object {
                [PSCustomObject]@{ fileId=$_.fileId; apiType=$_.apiType; error=$_.error; statusCode=$_.statusCode; timestamp=$_.timestamp }
            })
            failedFolders    = @($script:FailedFolders | ForEach-Object {
                [PSCustomObject]@{ folderId=$_.folderId; folderPath=$_.folderPath; error=$_.error; statusCode=$_.statusCode; timestamp=$_.timestamp }
            })
            bots             = $botArray
            folders          = $folderArray
        }

        $cpJson  = $cpData | ConvertTo-Json -Depth 8 -Compress
        $cpPath  = Join-Path (Get-Location).Path $script:CheckpointPath
        $tmpPath = "$cpPath.tmp"

        $cpJson | Out-File -FilePath $tmpPath -Encoding UTF8 -Force
        if (Test-Path $cpPath) { Remove-Item $cpPath -Force }
        Rename-Item -Path $tmpPath -NewName (Split-Path $cpPath -Leaf)

        Write-Log ("Checkpoint saved: $($Bots.Count) bots, " +
                   "$($script:CompletedFolders.Count) completed folders -> $($script:CheckpointPath)") -Level DEBUG
    } catch {
        Write-Log "WARNING: Could not save checkpoint: $($_.Exception.Message)" -Level WARN
    }
}


function Load-Checkpoint {
    <#
    .SYNOPSIS
        Reads and deserialises the checkpoint file.
        Returns $null if the file is absent or corrupt.
    #>
    $cpPath = Join-Path (Get-Location).Path $script:CheckpointPath
    if (-not (Test-Path $cpPath)) { return $null }

    try {
        $raw = Get-Content -Path $cpPath -Encoding UTF8 -Raw
        $cp  = $raw | ConvertFrom-Json
        if ($cp.version -ne "2.0") {
            Write-Log "WARNING: checkpoint version '$($cp.version)' is not '2.0'. Ignoring." -Level WARN
            return $null
        }
        return $cp
    } catch {
        Write-Log "WARNING: checkpoint file is corrupt and will be ignored: $($_.Exception.Message)" -Level WARN
        return $null
    }
}


function Restore-FromCheckpoint {
    <#
    .SYNOPSIS
        Applies a loaded checkpoint to the script's global state.
        Populates $Bots and $Folders (passed by reference as ArrayLists),
        rebuilds BotCache, and restores all row counters and failure lists.
    #>
    param(
        $CheckpointData,
        [System.Collections.ArrayList]$Bots,
        [System.Collections.ArrayList]$Folders
    )

    Write-Log "Restoring from checkpoint saved at $($CheckpointData.savedAt) ..."

    # ── Row counters ─────────────────────────────────────────────────────────
    $rc = $CheckpointData.rowCounters
    $script:Excel_BotRow   = [int]$rc.botRow
    $script:Excel_PkgRow   = [int]$rc.pkgRow
    $script:Excel_FolderRow = [int]$rc.folderRow
    $script:FullView_Row   = [int]$rc.fullViewRow
    $script:BatchCount     = [int]$rc.batchCount
    $script:BotCounter     = [int]$rc.botCounter
    $script:FolderCounter  = [int]$rc.folderCounter

    # ── Completed folder set ─────────────────────────────────────────────────
    foreach ($fid in $CheckpointData.completedFolders) {
        [void]$script:CompletedFolders.Add([string]$fid)
    }

    # ── Failure tracking ─────────────────────────────────────────────────────
    foreach ($f in $CheckpointData.failedApiCalls) {
        [void]$script:FailedApiCalls.Add([PSCustomObject]@{
            fileId     = [string]$f.fileId
            apiType    = [string]$f.apiType
            error      = [string]$f.error
            statusCode = [int]$f.statusCode
            timestamp  = [string]$f.timestamp
        })
    }
    foreach ($f in $CheckpointData.failedFolders) {
        [void]$script:FailedFolders.Add([PSCustomObject]@{
            folderId   = [string]$f.folderId
            folderPath = [string]$f.folderPath
            error      = [string]$f.error
            statusCode = [int]$f.statusCode
            timestamp  = [string]$f.timestamp
        })
    }

    # ── Bots ─────────────────────────────────────────────────────────────────
    foreach ($b in $CheckpointData.bots) {
        $botRecord = [PSCustomObject]@{
            id           = [string]$b.id
            name         = [string]$b.name
            path         = [string]$b.path
            parent_id    = [string]$b.parent_id
            depth        = [int]$b.depth
            children     = if ($b.children) {
                               [System.Collections.ArrayList]@(
                                   $b.children | ForEach-Object {
                                       [PSCustomObject]@{ id=[string]$_.id; name=[string]$_.name; path=[string]$_.path }
                                   }
                               )
                           } else { [System.Collections.ArrayList]@() }
            packages     = if ($b.packages) {
                               [System.Collections.ArrayList]@(
                                   $b.packages | ForEach-Object {
                                       [PSCustomObject]@{ name=[string]$_.name; version=[string]$_.version }
                                   }
                               )
                           } else { [System.Collections.ArrayList]@() }
            created_by   = [string]$b.created_by
            created_on   = [string]$b.created_on
            modified_by  = [string]$b.modified_by
            modified_on  = [string]$b.modified_on
            locked       = [bool]$b.locked
            file_size    = [long]$b.file_size
            dataComplete = [bool]$b.dataComplete
            dataIssues   = if ($b.dataIssues) {
                               [System.Collections.ArrayList]@($b.dataIssues | ForEach-Object { [string]$_ })
                           } else { [System.Collections.ArrayList]@() }
        }
        [void]$Bots.Add($botRecord)
        $script:BotCache[$botRecord.id] = $botRecord    # prevents duplicate API calls
        [void]$script:RestoredBotIds.Add($botRecord.id) # prevents duplicate Excel rows
    }

    # ── Folders ──────────────────────────────────────────────────────────────
    foreach ($f in $CheckpointData.folders) {
        $folderRecord = [PSCustomObject]@{
            id        = [string]$f.id
            name      = [string]$f.name
            path      = [string]$f.path
            parent_id = [string]$f.parent_id
            depth     = [int]$f.depth
        }
        [void]$Folders.Add($folderRecord)
        [void]$script:RestoredFolderIds.Add($folderRecord.id)
    }

    # Incremental mode must be on so Flush-FullViewSubtree fires for new bots
    $script:IncrementalMode = $true

    Write-Log ("Checkpoint restore complete -- $($Bots.Count) bots, " +
               "$($Folders.Count) folders, $($script:CompletedFolders.Count) completed folders")
    Write-Log ("Resuming Excel at row -- AllBots: $($script:Excel_BotRow), " +
               "PkgPerBot: $($script:Excel_PkgRow), " +
               "FolderStructure: $($script:Excel_FolderRow), " +
               "FullView: $($script:FullView_Row)")
}


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────
function Invoke-Main {

    # ── Pre-flight: validate config before ANY work ───────────────────────────
    Assert-Config

    # ── Initialise collections ────────────────────────────────────────────────
    $bots    = [System.Collections.ArrayList]@()
    $folders = [System.Collections.ArrayList]@()
    $visited = [System.Collections.Generic.HashSet[string]]@()

    # ── Checkpoint: check for an interrupted previous run ────────────────────
    $resuming = $false
    $cpData   = Load-Checkpoint
    if ($null -ne $cpData) {
        $cpDate    = try { [DateTime]::Parse($cpData.savedAt).ToString("dd MMM yyyy HH:mm:ss") } catch { $cpData.savedAt }
        $cpBots    = $cpData.botCount
        $cpFolders = $cpData.folderCount

        Write-Host ""
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
        Write-Host " CHECKPOINT FOUND" -ForegroundColor Yellow
        Write-Host "   Saved at : $cpDate" -ForegroundColor Yellow
        Write-Host "   Bots     : $cpBots" -ForegroundColor Yellow
        Write-Host "   Folders  : $cpFolders" -ForegroundColor Yellow
        Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Yellow
        Write-Host ""

        # Check the Excel file still exists -- if deleted we must start fresh.
        $outPath = Join-Path (Get-Location).Path $OUTPUT_FILE
        if (-not (Test-Path $outPath)) {
            Write-Host "WARNING: Output Excel file not found. Cannot resume -- starting fresh." -ForegroundColor Yellow
            $cpData = $null
        } else {
            $answer = Read-Host "Resume from checkpoint? [Y] Resume  [N] Start fresh"
            if ($answer -match '^[Yy]') {
                $resuming = $true
            } else {
                Write-Log "User chose fresh start -- deleting checkpoint."
                $cpPath = Join-Path (Get-Location).Path $script:CheckpointPath
                if (Test-Path $cpPath) { Remove-Item $cpPath -Force }
            }
        }
    }

    if ($resuming) {
        # Restore all state from checkpoint; skip Excel init (file already exists).
        Restore-FromCheckpoint -CheckpointData $cpData -Bots $bots -Folders $folders
        Write-Log "Resuming scan -- $($bots.Count) bots already collected."
    } else {
        # Fresh run.
        $script:BotCounter    = 0
        $script:FolderCounter = 0

        # Pre-flight: verify output file is writable before a long run
        Test-OutputFileLock

        # Create workbook up-front with all 9 sheets + incremental headers
        Initialize-IncrementalExcel
    }

    # ── Authenticate (always -- token may have expired since last run) ────────
    Invoke-AAAuthenticate

    # ── Walk the folder tree ──────────────────────────────────────────────────
    Write-Log "Scanning from folder id=$ROOT_FOLDER ..."
    Invoke-WalkFolder -FolderId $ROOT_FOLDER -FolderPath "/folder_$ROOT_FOLDER" -Depth 1 `
                      -Bots $bots -Folders $folders -SeenFolders $visited

    Write-Log "Scan complete -- $($bots.Count) bots, $($folders.Count) folders"

    # ── Flush any remaining partial batch ─────────────────────────────────────
    Write-Log "Flushing final incremental batch (if any remaining)..."
    Flush-IncrementalBatches
    Save-Checkpoint -Bots $bots -Folders $folders   # final checkpoint before aggregate writes

    # ── Build relationships ───────────────────────────────────────────────────
    $relationships = Build-Relationships -Bots $bots
    Write-Log "Relationships -- $($relationships.Count) parent->child links"

    # ── Generate report (aggregate sheets + Scan Issues) ─────────────────────
    Invoke-GenerateReport -Bots $bots -Folders $folders -Relationships $relationships

    # ── Print run summary to console ──────────────────────────────────────────
    Write-RunSummary -Bots $bots -Folders $folders

    # ── Delete checkpoint -- run completed successfully ───────────────────────
    $cpPath = Join-Path (Get-Location).Path $script:CheckpointPath
    if (Test-Path $cpPath) {
        Remove-Item $cpPath -Force
        Write-Log "Checkpoint deleted (run completed successfully)."
    }

    Write-Host "Done. Open: $OUTPUT_FILE" -ForegroundColor Green
    Write-Host ("   Sheets: Full View | Summary | Folder Structure | All Bots | " +
                "Bot Relationships | Hierarchy Tree | Packages Per Bot | Package Summary | Scan Issues") -ForegroundColor Green
}


# ── Entry point ──
try {
    Invoke-Main
} catch {
    Write-Host ""
    Write-Host "FATAL ERROR: $_" -ForegroundColor Red
    Write-Host "Stack trace:" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}
