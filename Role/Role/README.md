# Automation Anywhere 360 (v30) - Role & Directory Permission Report

A PowerShell-based tool to generate comprehensive reports of roles and their directory permissions in Automation Anywhere 360 Control Room.

## Features

- \*\*Roles List: All roles with basic information (ID, name, description, created/modified dates)
- \*\*Directory Permissions: Full recursive directory tree per role with all permissions
- \*\*Scan Issues: Log of any API failures encountered
- **Excel Output**: Professional Excel file with formatted worksheets

## Prerequisites

- Windows PowerShell 5.1 or later
- [ImportExcel](https://github.com/dfinke/ImportExcel) PowerShell module

### Install ImportExcel:

```powershell
Install-Module ImportExcel -Scope CurrentUser
```

## Setup & Configuration

1. Clone or download this repository to your local machine
2. Open `Role_Directory_Permission_Scanner.ps1
3. Configure the following settings at the top of the script (or use environment variables):

| Variable    | Description                                        |
| ----------- | -------------------------------------------------- |
| `$CR_URL`   | Your Control Room URL (must start with `https://`) |
| `$USERNAME` | Your Automation Anywhere username                  |
| `$PASSWORD` | Your Automation Anywhere password                  |

Or set as environment variables:

```powershell
$env:AA_CR_URL = "https://your-control-room-url"
$env:AA_USERNAME = "your-username"
$env:AA_PASSWORD = "your-password"
```

## Usage

### Run with PowerShell

```powershell
.\Role_Directory_Permission_Scanner.ps1
```

### Run with Batch File (Windows)

Double-click `Role_Directory_Permission.bat`

## Output

The script generates an Excel file `Role_Directory_Report.xlsx` with 4 sheets:

1. **Role IDs**: (Input sheet to filter roles to scan (optional)\*\*
2. **Roles**: List of all roles with counts and metadata
3. **Directories**: Full directory tree with permissions for each role
4. **Scan Issues**: Any API call failures during execution

## Configuration Options

| Setting               | Default | Description                      |
| --------------------- | ------- | -------------------------------- |
| `$PAGE_SIZE`          | 100     | Roles per page (max 100)         |
| `$DELAY_MS`           | 100     | Milliseconds between API calls   |
| `$VERIFY_SSL`         | `$true` | Verify SSL certificates          |
| `$MAX_RETRIES`        | 1       | Max retries for API calls        |
| `$TOKEN_LIFETIME_SEC` | 900     | Re-authenticate every 15 minutes |

## License

This project is provided as-is for internal use.
