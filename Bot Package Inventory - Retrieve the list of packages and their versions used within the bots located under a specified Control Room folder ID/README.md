# Bot Package Inventory - Retrieve the list of packages and their versions used within the bots located under a specified Control Room folder ID

## Purpose and when to use it

Control Room reports show which bots exist, but not which package versions each bot is built on. This inventory reads the content of every taskbot in a folder (including all subfolders, at any depth) and lists, for each bot, every package it declares, the version, and how many commands from that package the bot actually uses.

Use it when you need an inventory of package usage, for example before a package upgrade, or as the input to a vulnerability review.

The same solution is provided in two forms. Pick the one that fits your environment; both produce the same `PackageReport.xlsx`.

## What the two folders contain

| Folder | What it is | Use it when |
|---|---|---|
| `Python Solution` | A standalone Python script, `TracePackagesnCommandsScript.py`, that runs the scan from a machine with Python 3 (libraries `openpyxl` and `requests`). It can also be launched from an Automation Anywhere bot that runs a Python script. | You want a quick, no-install scan from a workstation or a server, or you already run Python in your pipelines. No Control Room package import is needed. |
| `AA custom package` | The `BotPackageInventory-1.0.1.jar` Automation Anywhere package. It exposes the **Trace Packages and Commands** action, which is dropped into a bot and runs on a Windows Bot Agent. | You want the scan to live in the Control Room as a governed, reusable action, run by bots on the Bot Agent with no Python or pip on the machine. |

Both read the same Control Room APIs, apply the same retries, write the same progress files for resume, and build the same `PackageReport.xlsx`. The only real difference is how you run them and where they execute.

## What they produce

`PackageReport.xlsx` in the output folder, with two sheets:

- **Full Report** - one row for each package in each bot: Bot Id, Bot Name, Bot Path, Package, Artifact Name, Version, Commands Used.
- **Bot Level Report** - one row for each bot, including bots that could not be read: Bot Id, Bot Name, Bot Path, Status, Package Count, Total Commands, Packages.

When the run ends, both return a short JSON summary:

```json
{
    "bots_read": 1250,
    "bots_failed": 3,
    "bots_skipped_from_earlier_run": 0,
    "folders_failed": 0,
    "report": "C:\\Reports\\PackageScan\\PackageReport.xlsx"
}
```

## Common requirements

- A Control Room user and API key that can see the folders and bots in scope and can view packages. Bots the account cannot read show up in the report with the status "Content not retrieved".
- Network access to the Control Room URL from the machine that runs the scan (the workstation for the Python script, the Bot Agent machine for the package).
- The ID of the folder to scan. The scan covers that folder and everything below it.
- An output folder for the report, the log and the progress files.
- Close `PackageReport.xlsx` in Excel before the run ends. If it is open, the report is saved under a new name with the time added.

## Getting started

1. Choose your folder above.
2. Open the `README.md` inside that folder for the full instructions (before you run it, how to run it, how it works, report contents, stopping and resuming, and the settings you can change).
3. Start with a small folder and a low `MAX_WORKERS` the first time, and watch the log.
