Monitoring the Control Room Device Status (Connected/Disconnected)

PowerShell implementation of the Automation Anywhere Device Monitor

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Configuration](#configuration)
4. [How to Run](#how-to-run)
5. [Key Features](#key-features)
6. [Code Structure](#code-structure)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This PowerShell script monitors device statuses in Automation Anywhere Control Room via v2 REST API and logs disconnected devices to a file.

---

## Prerequisites

- Windows PowerShell 5.1 or PowerShell 7+
- Network access to Automation Anywhere Control Room
- Valid Control Room credentials

---

## Configuration

Edit the following variables at the top of [monitor.ps1]

```powershell
# ==========================
# Configuration
# ==========================
$USERNAME = "ywz"        # Your Control Room username
$password = "pass"            # Your Control Room password
$CHECK_INTERVAL = 30              # Seconds between device checks
$logPath = "disconnected_devices.log"  # Log file path
$baseUrl = "YOUR_BASE_URL_HERE"  # Control Room URL
$hostnames = @()                  # Hostnames to filter (use @() to monitor ALL)
```

### Hostname Filtering

- Uses `substring` operator for OR matching
- Example: `@("n9k", "ywz")` matches any hostname containing "n9k" OR "ywz"
- Set to `@()` to monitor ALL devices

---

## How to Run

### Option 1: Launch via Batch File

Double-click [launch_monitor.bat]

### Option 2: Run Directly in PowerShell

1. Open PowerShell
2. Navigate to the script directory
3. Run:
   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File monitor.ps1
   ```

---

## Key Features

1. **Automatic Authentication**: Handles token acquisition
2. **Proactive Re-authentication**: Re-authenticates every 15 minutes (before token expiry)
3. **401 Fallback**: Re-authenticates if token expires unexpectedly
4. **Hostname Filtering**: Filter devices by hostname substring
5. **Logging**: Logs disconnected devices with timestamps to `disconnected_devices.log`

---

## Code Structure

### Class: `DeviceMonitor`

Encapsulates all monitoring logic:

- `Authenticate()`: Connects to Control Room and gets token
- `BuildPayload()`: Creates API request payload with optional hostname filter
- `CheckDevices()`: Checks device statuses and logs disconnected ones
- `Run()`: Main monitoring loop

---

## Troubleshooting

### Common Errors

#### 401 Unauthorized

- **Cause**: Token expired
- **Resolution**: Handled automatically

#### Authentication Failed

- **Cause**: Invalid credentials or network issue
- **Resolution**: Check username/password and network connectivity

#### Execution Policy Restricted

- **Cause**: PowerShell execution policy blocking script
- **Resolution**: Use `-ExecutionPolicy Bypass` flag when launching

---

## Log Format

Each log entry follows:

```
YYYY-MM-DD HH:mm:ss - Device {hostname} is DISCONNECTED
```
