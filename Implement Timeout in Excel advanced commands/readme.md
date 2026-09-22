# Excel Watchdog Solution - README

## Overview

Sometimes, while a bot is working with Excel using Automation Anywhere's Excel Advanced commands, Excel may stop responding. This can happen due to large files, long-running operations, or unexpected application issues.

When Excel becomes unresponsive, the bot may get stuck waiting for Excel to respond, causing the automation to fail or run indefinitely.

To avoid this situation, a watchdog solution has been implemented using a PowerShell script and a BAT file. The watchdog monitors a specific Excel workbook and automatically closes it if it remains unresponsive for a configured period of time.

The bot can then identify that Excel was closed by the watchdog and handle the error appropriately.

---

# Prerequisites

Before using this solution, make sure the following are available:

- Automation Anywhere Bot Creator license.
- A `.txt` file for storing watchdog logs.
- BAT file and PowerShell script deployed to the machine where the bot will run.

---

# Required Inputs

The bot requires the following inputs:

## Timeout Seconds

The amount of time the watchdog should wait before checking whether Excel is responding.

Example:

```text
120
```

---

## Excel File Path / File Name

The workbook that needs to be monitored.

Example:

```text
FreezeExcel_Workbook 1 - Excel
```

**Note:** Do not provide the file path within double quotes.

---

## Log File Path

The location of the text file used to store watchdog logs.

Example:

```text
C:\Logs\ExcelWatchdogLog.txt
```

**Note:** Do not provide the file path within double quotes.

---

## BAT File Path

The location of the BAT file that starts the PowerShell watchdog script.

---

# What the BAT File Does

The BAT file acts as a launcher for the PowerShell script.

Its responsibilities are:

- Accept input parameters from the bot.
- Validate that the required parameters are provided.
- Pass the parameters to the PowerShell script.
- Return the execution status back to the bot.

If the required inputs are not provided, the BAT file will stop and display an appropriate error message.

---

# What the PowerShell Script Does

The PowerShell script is responsible for monitoring Excel.

The script performs the following actions:

1. Receives the timeout value, workbook name, and log file path.
2. Waits for the configured number of seconds.
3. Checks for any Excel processes that are not responding.
4. Compares the Excel window title with the workbook name provided by the bot.
5. If a matching non-responsive Excel workbook is found:
   - The Excel process is terminated.
   - Details are written to the log file.
6. If no hung Excel process is found, the script completes without taking any action.

---

# How the Bot Works

## Step 1: Capture the Start Time

Before starting the watchdog, the bot captures the current timestamp.

This timestamp is used later to identify log entries generated during the current execution.

---

## Step 2: Start the Watchdog

Using the Application package, the bot executes the BAT file and passes:

- Timeout Seconds
- Excel File Name
- Log File Path

This starts the PowerShell watchdog in parallel.

---

## Step 3: Execute Excel Operations

The bot continues with its normal Excel processing using the Excel Advanced package.

For example:

- Opening workbooks
- Reading data
- Writing data
- Refreshing formulas
- Saving files

---

## Step 4: Monitor for Unresponsive Excel

While the bot is working with Excel, the watchdog runs in the background.

If Excel remains unresponsive for longer than the configured timeout period:

- The watchdog identifies the Excel process.
- The watchdog closes the Excel instance.
- The action is recorded in the log file.

---

## Step 5: Exception Handling

If Excel is closed by the watchdog, the current Excel action fails.

The bot then moves to the Catch block.

---

## Step 6: Read the Log File

Inside the Catch block, the bot reads the watchdog log file.

The bot only reviews log entries that were created after the timestamp captured in Step 1.

---

## Step 7: Check Whether the Watchdog Closed Excel

The bot searches the log file for an entry similar to:

```text
Successfully killed process PID
```

If this entry exists, the bot knows that the workbook became unresponsive and was terminated by the watchdog.

---

## Step 8: Perform Appropriate Error Handling

Based on the log entry, the bot can:

- Write a clear error message to the logs.
- Throw a custom exception.
- Trigger recovery or remediation steps.
- Notify support teams if required.

Example:

> Excel became unresponsive and was automatically terminated after exceeding the configured timeout period.

---

# Sample Log Entries

```text
[2025-08-15 10:00:00] Script started with parameters: WaitSeconds=120, FileName=FreezeExcel_Workbook 1 - Excel

[2025-08-15 10:00:00] Starting sleep for 120 seconds...

[2025-08-15 10:02:00] Sleep completed. Checking for non-responding Excel processes...

[2025-08-15 10:02:01] Found 1 non-responding Excel process(es)

[2025-08-15 10:02:02] Match found! Window title contains 'FreezeExcel_Workbook 1 - Excel'. Killing PID: 35864

[2025-08-15 10:02:02] Successfully killed process PID: 35864 | FileName: FreezeExcel_Workbook 1 - Excel
```

---

